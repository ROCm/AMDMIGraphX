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
#ifndef MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
#define MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/benchmark_candidate.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/* Generate an input argument for each parameter of the program in parameter
   order. Inputs found in fill_map are filled with the given value on the host;
   the rest are filled with random data generated on the GPU. */
MIGRAPHX_GPU_EXPORT std::vector<argument>
generate_program_arguments(const context& ictx,
                           const program& p,
                           const std::unordered_map<std::string, double>& fill_map = {});

/* Time each candidate and return the fastest one */
struct MIGRAPHX_GPU_EXPORT simple_benchmark
{
    int bundle = 1;
    int nruns  = 100;

    const benchmark_candidate& run(const context& ictx,
                                   const std::vector<benchmark_candidate>& candidates) const;
};

/* Time every candidate with a quick coarse measurement, then precisely re-time
   the top candidates with more iterations and return the fastest one */
struct MIGRAPHX_GPU_EXPORT adaptive_topk_benchmark
{
    // Number of successful precise timings to collect. Zero precisely times every candidate.
    std::size_t top_k = 10;
    // How much time to spend on precise timing
    std::size_t precise_ms         = 20;
    std::size_t coarse_ms          = 10;
    std::size_t precise_min_bundle = 4;
    std::size_t max_runs           = 20;

    const benchmark_candidate& run(const context& ictx,
                                   const std::vector<benchmark_candidate>& candidates) const;
};

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

/* benchmark gpu::code_object with expected input shapes over n iterations */
MIGRAPHX_GPU_EXPORT double
time_op(const context& ictx, operation op, int bundle = 1, int nruns = 100);

MIGRAPHX_GPU_EXPORT double
time_loop(migraphx::gpu::context& gctx, int bundle, int nruns, const std::function<void()>& f);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DRIVER_PERF_HPP
