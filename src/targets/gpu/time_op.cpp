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
#include <migraphx/time.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/hip.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_BUNDLE);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BENCHMARKING_NRUNS);

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

double
time_loop(migraphx::gpu::context& gctx, int bundle, int nruns, const std::function<void()>& f)
{
    return time_loop(gctx, bundle, nruns, f, true);
}

double time_loop(migraphx::gpu::context& gctx,
                 int bundle,
                 int nruns,
                 const std::function<void()>& f,
                 bool warmup)
{
    // check for manual overrides; value_of caches its result, so the fallback must not be a
    // per-call value
    if(auto b = value_of(MIGRAPHX_BENCHMARKING_BUNDLE{}, 0); b > 0)
        bundle = b;
    if(auto n = value_of(MIGRAPHX_BENCHMARKING_NRUNS{}, 0); n > 0)
        nruns = n;
    if(bundle <= 0 or nruns <= 0)
        MIGRAPHX_THROW("Timing bundle and runs must be greater than zero");

    std::vector<std::pair<hip_event_ptr, hip_event_ptr>> events(nruns);
    std::generate(events.begin(), events.end(), [] {
        return std::make_pair(context::create_event_for_timing(),
                              context::create_event_for_timing());
    });
    std::vector<double> times;
    times.reserve(nruns);
    if(warmup)
        f();
    for(auto i : range(nruns))
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
    std::transform(events.begin(), events.end(), std::back_inserter(times), [&](const auto& p) {
        return context::get_elapsed_ms(p.first.get(), p.second.get()) / bundle;
    });
    std::sort(times.begin(), times.end());

    // compute common average by removing top and bottom 25% of values
    std::size_t quarters = times.size() / 4;
    double total         = std::accumulate(times.begin() + quarters, times.end() - quarters, 0.0);
    return total / std::distance(times.begin() + quarters, times.end() - quarters);
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

std::vector<argument> generate_program_arguments(
    const context& ictx, const program& p, const std::unordered_map<std::string, double>& fill_map)
{
    std::unordered_map<std::string, argument> generated;
    return generate_program_arguments(ictx, p, fill_map, generated);
}

std::vector<argument>
generate_program_arguments(const context& ictx,
                           const program& p,
                           const std::unordered_map<std::string, double>& fill_map,
                           std::unordered_map<std::string, argument>& generated)
{
    auto gctx      = ictx;
    const auto* mm = p.get_main_module();
    auto names     = mm->get_parameter_names();
    std::vector<argument> args;
    args.reserve(names.size());
    unsigned long seed = 0;
    std::transform(names.begin(), names.end(), std::back_inserter(args), [&](const auto& name) {
        auto s         = mm->get_parameter_shape(name);
        std::string id = "";
        if(s.type() != migraphx::shape::tuple_type)
            id = s.type_string() + migraphx::shape::to_sizes_string({s.as_standard()});

        auto fill = fill_map.find(id);
        // Neither fill tag contains ':', so the first ':' ends the tag even if the name has one
        auto key = (fill == fill_map.end() ? std::string{"random"} : to_hex_float(fill->second)) +
                   ":" + name;
        auto& arg = generated[key];
        if(not arg.empty() and arg.get_shape() == s)
            return arg;
        // Release the stale argument first, so it and its replacement are never both resident
        arg = {};
        // fill_map inputs need specific values (host fill); the rest are generated
        // on the GPU to skip the host PRNG + H2D copy per candidate.
        if(fill != fill_map.end())
            arg = to_gpu(fill_argument(s, fill->second));
        else
            arg = gpu_generate_random(gctx, s, seed++);
        return arg;
    });
    return args;
}

parameter_map make_parameter_map(const_module_ref mod, const std::vector<argument>& args)
{
    auto names = mod->get_parameter_names();
    assert(names.size() == args.size());
    parameter_map param_map;
    std::transform(names.begin(),
                   names.end(),
                   args.begin(),
                   std::inserter(param_map, param_map.end()),
                   [](const auto& name, const auto& arg) { return std::make_pair(name, arg); });
    return param_map;
}

double time_program(const context& ictx,
                    program p,
                    const std::unordered_map<std::string, double>& fill_map,
                    int bundle,
                    int nruns)
{
    std::vector<migraphx::context> ctx_vec = {ictx};
    auto& gctx                             = any_cast<migraphx::gpu::context>(ctx_vec.front());
    p.get_main_module()->finalize(ctx_vec);
    auto param_map =
        make_parameter_map(p.get_main_module(), generate_program_arguments(ictx, p, fill_map));
    auto run = [&] { p.eval_with_context(ctx_vec, param_map); };
    return time_loop(gctx, bundle, nruns, run);
}

namespace {
struct benchmark_program
{
    program p;
    parameter_map param_map;

    double
    time(std::vector<migraphx::context>& ctx_vec, int bundle, int nruns, bool warmup = true) const
    {
        auto& gctx = any_cast<migraphx::gpu::context>(ctx_vec.front());
        return time_loop(
            gctx, bundle, nruns, [&] { p.eval_with_context(ctx_vec, param_map); }, warmup);
    }
};
} // namespace

// Pair the candidate's finalized program with newly generated arguments, building the program
// unless one is given
static benchmark_program make_benchmark_program(std::vector<migraphx::context>& ctx_vec,
                                                const benchmark_candidate& candidate,
                                                optional<program> finalized = nullopt)
{
    if(not finalized.has_value())
    {
        finalized = candidate.make_program();
        candidate.before_run(*finalized);
        finalized->get_main_module()->finalize(ctx_vec);
    }
    const auto& gctx = any_cast<migraphx::gpu::context>(ctx_vec.front());
    auto param_map   = make_parameter_map(finalized->get_main_module(),
                                        candidate.generate_arguments(gctx, *finalized));
    return {*std::move(finalized), std::move(param_map)};
}

const benchmark_candidate&
simple_benchmark::run(const context& ictx, const std::vector<benchmark_candidate>& candidates) const
{
    if(candidates.empty())
        MIGRAPHX_THROW("simple_benchmark: no candidates to benchmark");
    std::vector<migraphx::context> ctx_vec = {ictx};
    std::vector<double> times;
    times.reserve(candidates.size());
    std::transform(candidates.begin(),
                   candidates.end(),
                   std::back_inserter(times),
                   [&](const benchmark_candidate& candidate) {
                       auto trace = candidate.trace();
                       trace("Benchmarking solution: ", candidate.solution());
                       auto bp = make_benchmark_program(ctx_vec, candidate);
                       auto t  = bp.time(ctx_vec, bundle, nruns);
                       trace(t, "ms");
                       return t;
                   });
    auto fastest = std::min_element(times.begin(), times.end());
    return candidates.at(std::distance(times.begin(), fastest));
}

// Floor for measured times when sizing run counts and bundles; avoids division
// by zero for kernels that time near zero.
static constexpr double benchmark_min_time_ms = 1e-3;

static int compute_nruns(std::size_t budget_ms, double t, int bundle, std::size_t max_runs)
{
    double n = budget_ms / (std::max(t, benchmark_min_time_ms) * bundle);
    return std::clamp(n, 1.0, static_cast<double>(max_runs));
}

// Run the timing function f, tracing and mapping a failure to nullopt
template <class F>
static optional<double> try_benchmark(const tracer& trace, F f)
{
    try
    {
        return f();
    }
    catch(const std::exception& e)
    {
        trace("Benchmark failed: ", e.what());
    }
    catch(...)
    {
        trace("Benchmark failed");
    }
    return nullopt;
}

const benchmark_candidate&
adaptive_topk_benchmark::run(const context& ictx,
                             const std::vector<benchmark_candidate>& candidates) const
{
    if(candidates.empty())
        MIGRAPHX_THROW("adaptive_topk_benchmark: no candidates to benchmark");
    std::vector<migraphx::context> ctx_vec = {ictx};

    const double invalid = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> indices(candidates.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Coarse pass: warmup + single-run estimate, then a short bundle-of-1 measurement.
    // The second measurement is skipped when it cannot change who is precise-timed: the estimate
    // already exceeds coarse_ms (compute_nruns would return 1), or top_k finite times already
    // beat it. The accumulator is a max-heap, by coarse time, of the best top_k candidates so far;
    // their programs are kept for the precise pass and released when they drop out of the heap.
    // Their arguments are not kept: each set holds its candidate's scratch, which would otherwise
    // stay resident while later candidates allocate theirs.
    std::vector<double> coarse(candidates.size(), invalid);
    std::vector<optional<program>> kept(candidates.size());
    auto by_coarse = [&](auto i, auto j) { return coarse[i] < coarse[j]; };
    (void)std::accumulate(
        indices.begin(),
        indices.end(),
        std::vector<std::size_t>{},
        [&](std::vector<std::size_t> top, auto i) {
            const auto& candidate = candidates[i];
            auto trace            = candidate.trace();
            trace("Benchmarking solution: ", candidate.solution());
            auto t = try_benchmark(trace, [&] {
                auto bp                = make_benchmark_program(ctx_vec, candidate);
                auto estimate          = bp.time(ctx_vec, 1, 1);
                const bool over_budget = estimate > static_cast<double>(coarse_ms);
                const bool misses_top_k =
                    top_k > 0 and top.size() >= top_k and estimate > coarse[top.front()];
                double time = estimate;
                if(not over_budget and not misses_top_k)
                    time =
                        bp.time(ctx_vec, 1, compute_nruns(coarse_ms, estimate, 1, max_runs), false);
                if(top_k > 0)
                    kept[i] = std::move(bp.p);
                return time;
            });
            if(t.has_value())
                trace("Coarse time: ", *t, "ms");
            coarse[i] = t.value_or(invalid);
            if(top_k == 0 or not std::isfinite(coarse[i]))
                return top;
            if(top.size() < top_k)
            {
                top.push_back(i);
                std::push_heap(top.begin(), top.end(), by_coarse);
                return top;
            }
            auto dropped = i;
            if(coarse[i] < coarse[top.front()])
            {
                std::pop_heap(top.begin(), top.end(), by_coarse);
                dropped = std::exchange(top.back(), i);
                std::push_heap(top.begin(), top.end(), by_coarse);
            }
            kept[dropped] = nullopt;
            return top;
        });

    // Select the candidates that measured successfully, keep the top_k fastest
    std::vector<std::size_t> selected;
    selected.reserve(candidates.size());
    std::copy_if(indices.begin(), indices.end(), std::back_inserter(selected), [&](auto i) {
        return std::isfinite(coarse[i]);
    });
    if(selected.empty())
        MIGRAPHX_THROW("adaptive_topk_benchmark: all candidates failed to run");
    std::sort(
        selected.begin(), selected.end(), [&](auto i, auto j) { return coarse[i] < coarse[j]; });
    if(top_k > 0 and selected.size() > top_k)
        selected.resize(top_k);
    // Precise timing only separates close candidates, so one far behind the best coarse time
    // cannot win it. top_k == 0 asks for every candidate to be timed precisely.
    if(top_k > 0 and coarse_cutoff_factor > 0)
    {
        const double cutoff =
            coarse_cutoff_factor * std::max(coarse[selected.front()], benchmark_min_time_ms);
        selected.erase(std::upper_bound(selected.begin(),
                                        selected.end(),
                                        cutoff,
                                        [&](double c, auto i) { return c < coarse[i]; }),
                       selected.end());
    }

    // Pick one bundle for all precise runs, sized so the fastest candidate can
    // still fit max_runs measurements in the precise budget.
    double t_ref = coarse[selected.front()];
    int bundle = std::max<double>(precise_ms / (std::max(t_ref, benchmark_min_time_ms) * max_runs),
                                  precise_min_bundle);

    // Let the GPU cool down after the coarse pass so thermal throttling
    // doesn't skew the precise measurements
    std::this_thread::sleep_for(std::chrono::milliseconds{10});

    // Precise pass over the selected candidates, generating the arguments of one at a time. A kept
    // coarse program has already run on this stream, so only a rebuilt one needs a warmup.
    std::vector<double> precise(selected.size(), invalid);
    std::transform(selected.begin(), selected.end(), precise.begin(), [&](auto i) {
        const auto& candidate = candidates[i];
        auto trace            = candidate.trace();
        trace("Precise solution: ", candidate.solution());
        auto t = try_benchmark(trace, [&] {
            const bool rebuild = not kept[i].has_value();
            auto bp            = make_benchmark_program(ctx_vec, candidate, std::move(kept[i]));
            return bp.time(
                ctx_vec, bundle, compute_nruns(precise_ms, coarse[i], bundle, max_runs), rebuild);
        });
        if(t.has_value())
            trace("Precise time: ", *t, "ms");
        return t.value_or(invalid);
    });
    auto fastest = std::min_element(precise.begin(), precise.end());
    // Fall back to the best coarse candidate when every precise timing failed
    if(not std::isfinite(*fastest))
        return candidates.at(selected.front());
    return candidates.at(selected.at(std::distance(precise.begin(), fastest)));
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
