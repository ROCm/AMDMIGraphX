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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const onnx_tile_kernel = R"__migraphx__(
#include <migraphx/kernels/onnx_tile.hpp>
#include <args.hpp>
namespace migraphx {
extern "C" {
MIGRAPHX_GLOBAL void onnx_tile_kernel(void* in_data, void* output)
{
    make_tensors()(in_data, output)([](auto input, auto out) { onnx_tile(input, out); });
}
}
} // namespace migraphx
)__migraphx__";

struct onnx_tile_compiler : compiler<onnx_tile_compiler>
{
    std::vector<std::string> names() const { return {"tile"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        const std::size_t nelem =
            out_s.dynamic() ? out_s.element_space() : out_s.elements();
        options.set_launch_params(v, compute_global_for(ctx, nelem));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "onnx_tile_kernel";
        options.virtual_inputs = inputs;

        auto src = interpolate_string(onnx_tile_kernel, {});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
