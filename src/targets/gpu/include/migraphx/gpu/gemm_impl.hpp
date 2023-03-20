/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP

#include <iterator>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/time.hpp>

using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#if ROCBLAS_VERSION_MAJOR >= 2 && ROCBLAS_VERSION_MINOR >= 38
using flag_type = rocblas_gemm_flags;
#else
using flag_type = int;
#endif

/**
 * @brief Templated implementations of the compute() and finalize() methods of the Gemm operator.
 *        For each function there are overloads using either float or int32_t for the arguments
 * alpha and beta.
 *
 * @param ctx .
 * @param output_shape .
 * @param args .
 * @param alpha .
 * @param beta .
 * @param int8_x4_format .
 * @param compute_fp32 .
 */
void gemm_compute(context& ctx,
                  const shape& output_shape,
                  const std::vector<argument>& args,
                  float alpha,
                  float beta,
                  bool int8_x4_format,
                  bool compute_fp32,
                  int32_t solution_idx);

void gemm_compute(context& ctx,
                  const shape& output_shape,
                  const std::vector<argument>& args,
                  int32_t alpha,
                  int32_t beta,
                  bool int8_x4_format,
                  bool compute_fp32,
                  int32_t solution_idx);

int32_t gemm_finalize(context& ctx,
                      const shape& output_shape,
                      const std::vector<shape>& input_shapes,
                      float alpha,
                      float beta,
                      bool int8_x4_format,
                      bool compute_fp32);

int32_t gemm_finalize(context& ctx,
                      const shape& output_shape,
                      const std::vector<shape>& input_shapes,
                      int32_t alpha,
                      int32_t beta,
                      bool int8_x4_format,
                      bool compute_fp32,
                      int32_t solution_idx);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
