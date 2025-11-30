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
#ifndef MIGRAPHX_GUARD_GPU_GEN_CODEGEN_HPP
#define MIGRAPHX_GUARD_GPU_GEN_CODEGEN_HPP

#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/gpu/gen/export.h>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

namespace gen {

/// Generate kernel code for a pointwise module using gen IR passes
MIGRAPHX_GPU_GEN_EXPORT std::string generate_pointwise_kernel(const module& m,
                                                              const std::string& kernel_name);

/// Compile a gen IR program to a GPU code object operation
/// The program should contain lowered gen IR operations (vector_load, vector_store, etc.)
/// Parameters in the IR correspond to tensor_views in the generated kernel
/// Returns a code_object_op that can be run on the GPU
MIGRAPHX_GPU_GEN_EXPORT operation compile_gen(context& ctx,
                                              const program& p,
                                              const std::string& kernel_name = "gen_kernel");

/// Generate C++ code for a gen IR module
/// Used internally by compile_gen
MIGRAPHX_GPU_GEN_EXPORT std::string generate_gen_code(const module& m,
                                                      const std::string& kernel_name);

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_GEN_CODEGEN_HPP
