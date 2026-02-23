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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const copy_nd_kernel = R"__migraphx__(
#include <migraphx/kernels/copy_nd.hpp>
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void copy_nd_kernel(void* src, void* offsets, void* dest)
{
    make_tensors()(src, offsets, dest)([&](auto&&... xs) {
        copy_nd(xs..., MIGRAPHX_MAKE_CONSTANT(index_int{AXIS}));
    });
}

}

} // namespace migraphx
)__migraphx__";

struct copy_nd_compiler : compiler<copy_nd_compiler>
{
    std::vector<std::string> names() const { return {"copy_nd", "gpu::copy_nd"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(v.get("deref", false))
            MIGRAPHX_THROW("copy_nd: deref=true is not implemented for GPU");

        hip_compile_options options;
        const auto& src_shape = inputs[0];
        options.set_launch_params(v, compute_global_for(ctx, src_shape.elements()));
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "copy_nd_kernel";

        auto axis = v.at("axis").to<std::size_t>();
        options.emplace_param("-DAXIS=" + std::to_string(axis));

        return compile_hip_code_object(ctx, copy_nd_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
