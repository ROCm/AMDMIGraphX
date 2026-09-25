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
#include <tuple>
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
    // check for manual overrides
    bundle = value_of(MIGRAPHX_BENCHMARKING_BUNDLE{}, bundle);
    nruns  = value_of(MIGRAPHX_BENCHMARKING_NRUNS{}, nruns);
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

        // fill_map inputs need specific values (host fill); the rest are generated
        // on the GPU to skip the host PRNG + H2D copy per candidate.
        if(contains(fill_map, id))
            return to_gpu(fill_argument(s, fill_map.at(id)));
        return gpu_generate_random(gctx, s, seed++);
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

// Build the candidate's program and finalize it for ctx_vec
static program make_finalized_program(std::vector<migraphx::context>& ctx_vec,
                                      const benchmark_candidate& candidate)
{
    auto p = candidate.make_program();
    candidate.before_run(p);
    p.get_main_module()->finalize(ctx_vec);
    return p;
}

// Generate the arguments of the candidate's finalized program p. An argument in arg_cache is
// reused for a parameter with the same key and shape; one generated for a new shape replaces it,
// so arg_cache holds one argument per key.
static parameter_map make_benchmark_arguments(std::vector<migraphx::context>& ctx_vec,
                                              const benchmark_candidate& candidate,
                                              const program& p,
                                              std::unordered_map<std::string, argument>& arg_cache)
{
    auto& gctx     = any_cast<migraphx::gpu::context>(ctx_vec.front());
    const auto* mm = p.get_main_module();
    auto keys      = candidate.generate_argument_keys(p);
    if(keys.size() != mm->get_parameter_names().size())
        MIGRAPHX_THROW("benchmark_candidate: generate_argument_keys must give one key per "
                       "parameter");
    std::vector<argument> args;
    args.reserve(keys.size());
    std::transform(keys.begin(), keys.end(), std::back_inserter(args), [&](const auto& key_shape) {
        const auto& [key, s] = key_shape;
        auto& arg            = arg_cache[key];
        if(not arg.empty() and arg.get_shape() == s)
            return arg;
        // Release the stale argument first, so it and its replacement are never both resident
        arg = {};
        arg = candidate.generate_argument(gctx, key, s);
        return arg;
    });
    return make_parameter_map(mm, args);
}

static double time_benchmark(std::vector<migraphx::context>& ctx_vec,
                             const program& p,
                             const parameter_map& param_map,
                             int bundle,
                             int nruns,
                             bool warmup = true)
{
    auto& gctx = any_cast<migraphx::gpu::context>(ctx_vec.front());
    return time_loop(gctx, bundle, nruns, [&] { p.eval_with_context(ctx_vec, param_map); }, warmup);
}

const benchmark_candidate&
simple_benchmark::run(const context& ictx, const std::vector<benchmark_candidate>& candidates) const
{
    if(candidates.empty())
        MIGRAPHX_THROW("simple_benchmark: no candidates to benchmark");
    std::vector<migraphx::context> ctx_vec = {ictx};
    // The candidates are alternatives for the same computation, so they can share inputs
    std::unordered_map<std::string, argument> arg_cache;
    std::vector<double> times;
    times.reserve(candidates.size());
    std::transform(candidates.begin(),
                   candidates.end(),
                   std::back_inserter(times),
                   [&](const benchmark_candidate& candidate) {
                       auto trace = candidate.trace();
                       trace("Benchmarking solution: ", candidate.solution());
                       auto p         = make_finalized_program(ctx_vec, candidate);
                       auto param_map = make_benchmark_arguments(ctx_vec, candidate, p, arg_cache);
                       auto t         = time_benchmark(ctx_vec, p, param_map, bundle, nruns);
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

// Call f, tracing and mapping a failure to nullopt
template <class F>
static auto try_benchmark(const tracer& trace, F f) -> optional<decltype(f())>
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
    if(max_runs == 0 or coarse_max_runs == 0)
        MIGRAPHX_THROW("adaptive_topk_benchmark: max_runs and coarse_max_runs must be at least 1");
    std::vector<migraphx::context> ctx_vec = {ictx};
    // The candidates are alternatives for the same computation, so they can share inputs
    std::unordered_map<std::string, argument> arg_cache;

    const double invalid = std::numeric_limits<double>::infinity();

    // Build every program first and hold them through both passes. A candidate whose program
    // fails to build is skipped. Arguments are not held: each set holds its candidate's scratch,
    // which would otherwise stay resident while later candidates allocate theirs.
    std::vector<optional<program>> programs;
    programs.reserve(candidates.size());
    std::transform(candidates.begin(),
                   candidates.end(),
                   std::back_inserter(programs),
                   [&](const benchmark_candidate& candidate) {
                       auto trace = candidate.trace();
                       trace("Building solution: ", candidate.solution());
                       return try_benchmark(
                           trace, [&] { return make_finalized_program(ctx_vec, candidate); });
                   });

    // Coarse pass: warmup + single-run estimate, then a bundle-of-1 measurement of up to
    // coarse_max_runs runs within coarse_ms. The measurement is skipped when it would be a single
<<<<<<< HEAD
    // run too, so with the default coarse_max_runs every candidate is ranked by its estimate.
=======
    // run too, and the candidate is ranked by its estimate. The programs of the top_k fastest
    // candidates so far are kept for the precise pass, and released as soon as a faster one pushes
    // them out. Their arguments are not kept: each set holds its candidate's scratch, which would
    // otherwise stay resident while later candidates allocate theirs.
>>>>>>> 564522fe6d98bbaaa32e0f8d2559c033c9dfd592
    std::vector<double> coarse(candidates.size(), invalid);
    std::transform(candidates.begin(),
                   candidates.end(),
                   programs.begin(),
                   coarse.begin(),
                   [&](const benchmark_candidate& candidate, const optional<program>& p) {
                       if(not p.has_value())
                           return invalid;
                       auto trace = candidate.trace();
                       trace("Benchmarking solution: ", candidate.solution());
                       auto t = try_benchmark(trace, [&] {
                           auto param_map =
                               make_benchmark_arguments(ctx_vec, candidate, *p, arg_cache);
                           auto estimate = time_benchmark(ctx_vec, *p, param_map, 1, 1);
                           auto nruns    = compute_nruns(coarse_ms, estimate, 1, coarse_max_runs);
                           if(nruns == 1)
                               return estimate;
                           return time_benchmark(ctx_vec, *p, param_map, 1, nruns, false);
                       });
                       if(t.has_value())
                           trace("Coarse time: ", *t, "ms");
                       return t.value_or(invalid);
                   });

    // Select the candidates that measured successfully, keep the top_k fastest
    std::vector<std::size_t> indices(candidates.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<std::size_t> selected;
    selected.reserve(candidates.size());
    std::copy_if(indices.begin(), indices.end(), std::back_inserter(selected), [&](auto i) {
        return std::isfinite(coarse[i]);
    });
    if(selected.empty())
        MIGRAPHX_THROW("adaptive_topk_benchmark: all candidates failed to run");
    // Ties go to the lower index so the selection is deterministic
    std::sort(selected.begin(), selected.end(), [&](auto i, auto j) {
        return std::tie(coarse[i], i) < std::tie(coarse[j], j);
    });
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

    // Precise pass over the selected candidates, generating the arguments of one at a time. Each
    // program has already run on this stream in the coarse pass, so no warmup is needed.
    std::vector<double> precise(selected.size(), invalid);
    std::transform(selected.begin(), selected.end(), precise.begin(), [&](auto i) {
        const auto& candidate = candidates[i];
        const auto& p         = *programs[i];
        auto trace            = candidate.trace();
        trace("Precise solution: ", candidate.solution());
        auto t = try_benchmark(trace, [&] {
            auto param_map = make_benchmark_arguments(ctx_vec, candidate, p, arg_cache);
            auto nruns     = compute_nruns(precise_ms, coarse[i], bundle, max_runs);
            return time_benchmark(ctx_vec, p, param_map, bundle, nruns, false);
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
