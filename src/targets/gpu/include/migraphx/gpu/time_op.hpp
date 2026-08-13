/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
#define MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/optional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct adaptive_time_options
{
    std::size_t target_ms        = 100;
    std::size_t preferred_bundle = 1;
    // Floor on the sample count, taken even when it overruns target_ms, so that a candidate slower
    // than the budget is still measured enough times to reject an interfered run.
    std::size_t min_samples     = 4;
    std::size_t max_samples     = 20;
    std::size_t max_executions  = 10000;
    std::size_t warmup_ms       = 25;
    std::size_t max_warmup_runs = 50;
    std::size_t estimate_runs   = 1;
    double estimated_ms         = 0.0;
};

struct timing_schedule
{
    std::size_t bundle      = 1;
    std::size_t samples     = 1;
    std::size_t executions  = 1;
    std::size_t warmup_runs = 1;
};

struct adaptive_tuning_options
{
    adaptive_tuning_options()
    {
        coarse.target_ms = 10;
        coarse.warmup_ms = 5;
        // The coarse stage only has to rank candidates and it runs on every one of them, so it
        // keeps a single sample for slow candidates instead of paying the min_samples floor.
        coarse.min_samples    = 1;
        precise.estimate_runs = 5;
    }

    // Number of successful precise timings to collect. Zero precisely times every candidate.
    std::size_t top_k = 10;
    adaptive_time_options coarse{};
    adaptive_time_options precise{};
    // Delay after each candidate-stage timing to reduce thermal interference.
    std::size_t sleep_us = 100;
};

using adaptive_time_callback =
    std::function<optional<double>(std::size_t, const adaptive_time_options&)>;

MIGRAPHX_GPU_EXPORT timing_schedule make_timing_schedule(double estimate_ms,
                                                         const adaptive_time_options& options);

MIGRAPHX_GPU_EXPORT double adaptive_time_loop(migraphx::gpu::context& gctx,
                                              const adaptive_time_options& options,
                                              const std::function<void()>& f);

MIGRAPHX_GPU_EXPORT optional<std::size_t>
adaptive_time_topk(std::size_t candidate_count,
                   const adaptive_tuning_options& options,
                   const adaptive_time_callback& benchmark);

MIGRAPHX_GPU_EXPORT double time_op(const context& ictx,
                                   operation op,
                                   const std::vector<shape>& inputs,
                                   int bundle = 1,
                                   int nruns  = 100);

MIGRAPHX_GPU_EXPORT double time_program(const context& ictx,
                                        program p,
                                        const std::unordered_map<std::string, double>& fill_map,
                                        int bundle = 1,
                                        int nruns  = 100);

MIGRAPHX_GPU_EXPORT double
adaptive_time_program(const context& ictx,
                      program p,
                      const std::unordered_map<std::string, double>& fill_map,
                      const adaptive_time_options& options);

/* benchmark gpu::code_object with expected input shapes over n iterations */
MIGRAPHX_GPU_EXPORT double
time_op(const context& ictx, operation op, int bundle = 1, int nruns = 100);

MIGRAPHX_GPU_EXPORT double
time_loop(migraphx::gpu::context& gctx, int bundle, int nruns, const std::function<void()>& f);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
