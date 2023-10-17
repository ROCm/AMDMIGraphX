/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_MLIR_HPP

#include <string>
#include <vector>
#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/gpu/tuning_config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;
namespace gpu {

MIGRAPHX_GPU_EXPORT std::string dump_mlir(const module& m);
MIGRAPHX_GPU_EXPORT code_object_op compile_mlir(const context& migraphx_ctx,
                                                module m,
                                                const std::vector<instruction_ref>& inputs,
                                                const value& solution);

MIGRAPHX_GPU_EXPORT instruction_ref insert_mlir(module& m,
                                                instruction_ref ins,
                                                code_object_op co,
                                                const std::vector<instruction_ref>& inputs);

MIGRAPHX_GPU_EXPORT tuning_config get_tuning_config_mlir(const context& migraphx_ctx,
                                                         module m,
                                                         const std::vector<shape>& inputs,
                                                         bool exhaustive);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
