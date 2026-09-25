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
//
// te.py DSL for migraphx::gpu::benchmark_candidate.
//
// The generated header lives at
// src/targets/gpu/include/migraphx/gpu/benchmark_candidate.hpp; regenerate it
// with `cd tools && python generate.py` (generate_all routes include/gpu/ inputs
// into the gpu target tree). Do not edit the generated header by hand.
//
#ifndef MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP
#define MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/tracer.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/export.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifdef DOXYGEN

/// Type-erased interface for a tuning candidate that can be timed by a
/// benchmarker (see simple_benchmark and adaptive_topk_benchmark in
/// <migraphx/gpu/time_op.hpp>). A candidate knows how to build a runnable
/// program for itself and how to generate the input data used to run it.
struct benchmark_candidate
{
    /// Key and shape of the argument for each parameter of p, a program
    /// returned by make_program(), in parameter order. The benchmarker reuses
    /// a generated argument for any parameter, of this or another candidate,
    /// with the same key and shape, so equal keys must describe the same data.
    std::vector<std::pair<std::string, shape>> generate_argument_keys(const program& p) const;

    /// Generate the argument for a key and shape returned by
    /// generate_argument_keys().
    argument generate_argument(const context& ictx, const std::string& key, const shape& s) const;

    /// Build a runnable program for this candidate.
    program make_program() const;

    /// Tracer used to report benchmark progress; return a disabled tracer to
    /// silence the output.
    tracer trace() const;

    /// The tuning solution this candidate was compiled with.
    value solution() const;

    /// Called with the program before it is timed; used to print the program
    /// and other info on higher trace levels.
    void before_run(const program& p) const;
};

#else

<%
    interface('benchmark_candidate',
              virtual('generate_argument_keys',
                      returns = 'std::vector<std::pair<std::string, shape>>',
                      p       = 'const program&',
                      const   = True),
              virtual('generate_argument',
                      returns = 'argument',
                      ictx    = 'const context&',
                      key     = 'const std::string&',
                      s       = 'const shape&',
                      const   = True),
              virtual('make_program', returns = 'program', const = True),
              virtual('trace', returns = 'tracer', const = True),
              virtual('solution', returns = 'value', const = True),
              virtual('before_run', returns = 'void', p = 'const program&', const = True))
%>

#endif

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_BENCHMARK_CANDIDATE_HPP
