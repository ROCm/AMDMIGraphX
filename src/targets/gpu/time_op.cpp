/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/program.hpp>
#include <migraphx/gpu/time_op.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/context.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/hip.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <tuple>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_BUNDLE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_NRUNS);

constexpr double min_execution_ms = 0.001;

static std::vector<argument> generate_arguments(const std::vector<shape>& shapes,
                                                unsigned long seed = 0,
                                                random_mode rm     = random_mode::random)
{
    std::vector<argument> args;
    std::transform(shapes.begin(), shapes.end(), std::back_inserter(args), [&](const auto& s) {
        return to_gpu(generate_argument(s, seed++, rm));
    });
    return args;
}

static double common_average(std::vector<double> times)
{
    std::sort(times.begin(), times.end());
    const std::size_t quarters = times.size() / 4;
    // With fewer than four samples there is nothing to trim, and an untrimmed mean lets a single
    // interfered run dominate. Interference can only slow a run down, so use the fastest sample.
    if(quarters == 0)
        return times.front();
    const auto first = times.begin() + quarters;
    const auto last  = times.end() - quarters;
    return std::accumulate(first, last, 0.0) / std::distance(first, last);
}

template <class Env>
static optional<std::size_t> benchmarking_override(Env e)
{
    const auto v = value_of(e, 0);
    if(v == 0)
        return nullopt;
    return v;
}

static void validate_options(const adaptive_time_options& options)
{
    if(options.target_ms == 0)
        MIGRAPHX_THROW("Adaptive timing target must be greater than zero");
    if(options.preferred_bundle == 0)
        MIGRAPHX_THROW("Adaptive timing bundle must be greater than zero");
    if(options.min_samples == 0 or options.max_samples == 0)
        MIGRAPHX_THROW("Adaptive timing samples must be greater than zero");
    if(options.max_executions == 0)
        MIGRAPHX_THROW("Adaptive timing executions must be greater than zero");
    if(options.warmup_ms > 0 and options.max_warmup_runs == 0)
        MIGRAPHX_THROW("Adaptive timing warmup runs must be greater than zero");
}

// max_samples is settable from the outside while min_samples is not, so lowering only max_samples
// must clamp the floor rather than be rejected as an inconsistent pair.
static adaptive_time_options normalize_options(adaptive_time_options options)
{
    validate_options(options);
    options.min_samples = std::min(options.min_samples, options.max_samples);
    return options;
}

static adaptive_time_options resolve_options(adaptive_time_options options)
{
    if(const auto bundle = benchmarking_override(MIGRAPHX_BENCHMARKING_BUNDLE{}))
        options.preferred_bundle = *bundle;
    if(const auto samples = benchmarking_override(MIGRAPHX_BENCHMARKING_NRUNS{}))
        options.max_samples = *samples;
    return normalize_options(options);
}

static std::vector<double> measure_loop(migraphx::gpu::context& gctx,
                                        std::size_t bundle,
                                        std::size_t samples,
                                        const std::function<void()>& f)
{
    std::vector<std::pair<hip_event_ptr, hip_event_ptr>> events(samples);
    std::generate(events.begin(), events.end(), [] {
        return std::make_pair(context::create_event_for_timing(),
                              context::create_event_for_timing());
    });
    for(auto i : range(samples))
    {
        gctx.get_stream().record(events[i].first.get());
        for(auto j : range(bundle))
        {
            (void)j;
            f();
        }
        gctx.get_stream().record(events[i].second.get());
    }
    gctx.finish();
    std::vector<double> times;
    times.reserve(samples);
    std::transform(events.begin(), events.end(), std::back_inserter(times), [&](const auto& p) {
        return context::get_elapsed_ms(p.first.get(), p.second.get()) / bundle;
    });
    return times;
}

static double
estimate_time(migraphx::gpu::context& gctx, std::size_t nruns, const std::function<void()>& f)
{
    const auto times = measure_loop(gctx, nruns, 1, f);
    return std::max(times.front(), min_execution_ms);
}

timing_schedule make_timing_schedule(double estimate_ms, const adaptive_time_options& input_options)
{
    const auto options = normalize_options(input_options);
    if(not std::isfinite(estimate_ms) or estimate_ms <= 0.0)
        MIGRAPHX_THROW("Adaptive timing estimate must be finite and greater than zero");
    estimate_ms = std::max(estimate_ms, min_execution_ms);

    const double requested_executions = std::floor(options.target_ms / estimate_ms);
    const auto executions =
        requested_executions >= options.max_executions
            ? options.max_executions
            : std::max<std::size_t>(1, static_cast<std::size_t>(requested_executions));
    // min_samples is a floor rather than another budget cap: a candidate slower than target_ms
    // would otherwise be measured once, leaving common_average nothing to reject noise with.
    const auto preferred_samples = std::max<std::size_t>(1, executions / options.preferred_bundle);
    const auto samples =
        std::min(options.max_samples, std::max(options.min_samples, preferred_samples));
    const auto bundle = std::max<std::size_t>(1, executions / samples);

    std::size_t warmup_runs = 0;
    if(options.warmup_ms > 0)
    {
        const double requested_warmup_runs = std::ceil(options.warmup_ms / estimate_ms);
        warmup_runs =
            requested_warmup_runs >= options.max_warmup_runs
                ? options.max_warmup_runs
                : std::max<std::size_t>(1, static_cast<std::size_t>(requested_warmup_runs));
    }
    return {bundle, samples, bundle * samples, warmup_runs};
}

double adaptive_time_loop(migraphx::gpu::context& gctx,
                          const adaptive_time_options& input_options,
                          const std::function<void()>& f)
{
    const auto options = resolve_options(input_options);
    if(options.estimated_ms <= 0.0 and options.estimate_runs == 0)
        MIGRAPHX_THROW("Adaptive timing estimate runs must be greater than zero");

    // Run once to initialize lazy GPU resources and count it toward the warmup budget.
    f();
    std::size_t completed_warmup_runs = 1;

    double estimate_ms = options.estimated_ms;
    if(estimate_ms <= 0.0)
    {
        estimate_ms = estimate_time(gctx, options.estimate_runs, f);
        completed_warmup_runs += options.estimate_runs;
    }
    estimate_ms = std::max(estimate_ms, min_execution_ms);

    const auto schedule = make_timing_schedule(estimate_ms, options);
    if(schedule.warmup_runs > completed_warmup_runs)
    {
        const auto additional_warmup_runs = schedule.warmup_runs - completed_warmup_runs;
        for(auto i : range(additional_warmup_runs))
        {
            (void)i;
            f();
        }
    }

    auto times = measure_loop(gctx, schedule.bundle, schedule.samples, f);
    return common_average(std::move(times));
}

static optional<double> time_candidate(std::size_t i,
                                       const adaptive_time_options& options,
                                       std::size_t candidate_delay_us,
                                       const adaptive_time_callback& benchmark)
{
    auto result = benchmark(i, options);
    if(result.has_value() and (not std::isfinite(*result) or *result <= 0.0))
        result = nullopt;
    if(candidate_delay_us > 0)
        std::this_thread::sleep_for(std::chrono::microseconds{
            static_cast<std::chrono::microseconds::rep>(candidate_delay_us)});
    return result;
}

optional<std::size_t> adaptive_time_topk(std::size_t candidate_count,
                                         const adaptive_tuning_options& options,
                                         const adaptive_time_callback& benchmark)
{
    using delay_rep = std::chrono::microseconds::rep;
    if(options.sleep_us > static_cast<std::size_t>(std::numeric_limits<delay_rep>::max()))
        MIGRAPHX_THROW("Adaptive tuning candidate delay is too large");
    const auto coarse_options  = normalize_options(options.coarse);
    const auto precise_options = normalize_options(options.precise);

    std::vector<optional<double>> coarse(candidate_count);
    std::vector<optional<double>> precise(candidate_count);
    auto indices = range(candidate_count);

    if(options.top_k == 0)
    {
        std::transform(indices.begin(), indices.end(), precise.begin(), [&](auto i) {
            return time_candidate(i, precise_options, options.sleep_us, benchmark);
        });
    }
    else
    {
        std::transform(indices.begin(), indices.end(), coarse.begin(), [&](auto i) {
            return time_candidate(i, coarse_options, options.sleep_us, benchmark);
        });

        std::vector<std::size_t> ranked;
        std::copy_if(indices.begin(), indices.end(), std::back_inserter(ranked), [&](auto i) {
            return coarse[i].has_value();
        });
        std::sort(ranked.begin(), ranked.end(), [&](auto x, auto y) {
            const auto xtime = *coarse[x];
            const auto ytime = *coarse[y];
            return std::tie(xtime, x) < std::tie(ytime, y);
        });

        const auto top_k = std::min(options.top_k, ranked.size());
        std::accumulate(ranked.begin(), ranked.end(), std::size_t{0}, [&](auto measured, auto i) {
            if(measured >= top_k)
                return measured;
            auto candidate_options         = precise_options;
            candidate_options.estimated_ms = *coarse[i];
            precise[i] = time_candidate(i, candidate_options, options.sleep_us, benchmark);
            return measured + precise[i].has_value();
        });
    }

    std::vector<std::size_t> measured;
    std::copy_if(indices.begin(), indices.end(), std::back_inserter(measured), [&](auto i) {
        return precise[i].has_value();
    });
    if(measured.empty())
        return nullopt;
    return *std::min_element(measured.begin(), measured.end(), [&](auto x, auto y) {
        const auto xtime = *precise[x];
        const auto ytime = *precise[y];
        return std::tie(xtime, x) < std::tie(ytime, y);
    });
}

double
time_loop(migraphx::gpu::context& gctx, int bundle, int nruns, const std::function<void()>& f)
{
    if(bundle <= 0 or nruns <= 0)
        MIGRAPHX_THROW("Timing bundle and runs must be greater than zero");
    auto repeats = static_cast<std::size_t>(bundle);
    auto samples = static_cast<std::size_t>(nruns);
    if(const auto env_bundle = benchmarking_override(MIGRAPHX_BENCHMARKING_BUNDLE{}))
        repeats = *env_bundle;
    if(const auto env_samples = benchmarking_override(MIGRAPHX_BENCHMARKING_NRUNS{}))
        samples = *env_samples;
    f();
    return common_average(measure_loop(gctx, repeats, samples, f));
}

double
time_op(const context& ictx, operation op, const std::vector<shape>& inputs, int bundle, int nruns)
{
    // TODO: Use std::ref
    migraphx::context ctx = ictx;
    auto& gctx            = any_cast<migraphx::gpu::context>(ctx);
    auto output           = op.compute_shape(inputs);
    op.finalize(ctx, output, inputs);
    auto args = generate_arguments(inputs);
    auto run  = [&] { op.compute(ctx, output, args); };
    return time_loop(gctx, bundle, nruns, run);
}

double time_op(const context& ictx, operation op, int bundle, int nruns)
{
    auto inputs = any_cast<migraphx::gpu::code_object_op>(op).expected_inputs;
    return time_op(ictx, op, inputs, bundle, nruns);
}

template <class F>
static auto time_program_impl(const context& ictx,
                              program p,
                              const std::unordered_map<std::string, double>& fill_map,
                              F f)
{
    std::vector<migraphx::context> contexts = {ictx};
    auto& gctx                              = any_cast<migraphx::gpu::context>(contexts.front());
    auto* mm                                = p.get_main_module();
    mm->finalize(contexts);
    auto in_shapes = p.get_parameter_shapes();
    parameter_map params;
    unsigned long seed = 0;
    for(const auto& [name, shape] : in_shapes)
    {
        std::string id = "";
        if(shape.type() != migraphx::shape::tuple_type)
            id = shape.type_string() + migraphx::shape::to_sizes_string({shape.as_standard()});

        // fill_map inputs need specific values (host fill); the rest are generated
        // on the GPU to skip the host PRNG + H2D copy per candidate.
        if(contains(fill_map, id))
        {
            params[name] = to_gpu(fill_argument(shape, fill_map.at(id)));
        }
        else
        {
            params[name] = gpu_generate_random(gctx, shape, seed++);
        }
    }
    auto run = [&] { p.eval_with_context(contexts, params); };
    return f(gctx, run);
}

double time_program(const context& ictx,
                    program p,
                    const std::unordered_map<std::string, double>& fill_map,
                    int bundle,
                    int nruns)
{
    return time_program_impl(ictx, std::move(p), fill_map, [&](auto& gctx, const auto& run) {
        return time_loop(gctx, bundle, nruns, run);
    });
}

double adaptive_time_program(const context& ictx,
                             program p,
                             const std::unordered_map<std::string, double>& fill_map,
                             const adaptive_time_options& options)
{
    return time_program_impl(ictx, std::move(p), fill_map, [&](auto& gctx, const auto& run) {
        return adaptive_time_loop(gctx, options, run);
    });
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
