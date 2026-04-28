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
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/amdgpu_kernel_registry.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct amdgpu_gemm_add_compiler : compiler<amdgpu_gemm_add_compiler>
{
    std::vector<std::string> names() const { return {"amdgpu::gemm_add"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        (void)ctx;
        (void)inputs;
        (void)v;
        MIGRAPHX_THROW("amdgpu::gemm_add compile_op is not supported directly");
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v      = op.to_value();
        auto shapes = to_shapes(ins->inputs());
        auto match  = find_amdgpu_kernel(ctx, op, shapes);
        if(not match.found)
            MIGRAPHX_THROW(op.name() + ": no matching kernel in AMDGPU registry");

        value::binary bin{read_buffer(match.binary_path)};
        auto out_shape     = shapes.back();
        std::size_t local  = std::max<std::size_t>(1, match.local_size);
        std::size_t global = ((out_shape.elements() + local - 1) / local) * local;
        (void)ctx;
        (void)v;

        operation co = code_object_op{
            std::move(bin), match.symbol_name, global, local, shapes, out_shape, -1};
        return compiler_replace{co};
    }
};

struct amdgpu_gemm_gemm_add_compiler : compiler<amdgpu_gemm_gemm_add_compiler>
{
    std::vector<std::string> names() const { return {"amdgpu::gemm_gemm_add"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        (void)ctx;
        (void)inputs;
        (void)v;
        MIGRAPHX_THROW("amdgpu::gemm_gemm_add compile_op is not supported directly");
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v      = op.to_value();
        auto shapes = to_shapes(ins->inputs());
        auto match  = find_amdgpu_kernel(ctx, op, shapes);
        if(not match.found)
            MIGRAPHX_THROW(op.name() + ": no matching kernel in AMDGPU registry");

        value::binary bin{read_buffer(match.binary_path)};
        auto out_shape     = shapes.back();
        std::size_t local  = std::max<std::size_t>(1, match.local_size);
        std::size_t global = ((out_shape.elements() + local - 1) / local) * local;
        (void)ctx;
        (void)v;

        operation co = code_object_op{
            std::move(bin), match.symbol_name, global, local, shapes, out_shape, -1};
        return compiler_replace{co};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
