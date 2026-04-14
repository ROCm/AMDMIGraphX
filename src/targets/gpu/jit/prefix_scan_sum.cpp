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

static const char* const prefix_scan_sum_kernel = R"__migraphx__(
#include <migraphx/kernels/prefix_scan_sum.hpp>
#include <migraphx/kernels/slice.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void prefix_scan_sum_kernel(void* input_p, void* output_p)
{
    make_tensors()(input_p, output_p)([](auto input, auto output) {
        auto idx = make_index();
        slice_schedule<per_block>(idx, slice_axes<${axis}>())(input, output)(
            [&](auto in_slice, auto out_slice) {
                prefix_scan_sum_slice<${exclusive}, ${reverse}>(in_slice, out_slice);
            });
    });
}

}

} // namespace migraphx

)__migraphx__";

struct prefix_scan_sum_compiler : compiler<prefix_scan_sum_compiler>
{
    std::vector<std::string> names() const { return {"prefix_scan_sum"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = inputs;
        options.kernel_name    = "prefix_scan_sum_kernel";

        const auto& output_shape = inputs.back();
        auto exclusive           = v.get("exclusive", false);
        auto reverse             = v.get("reverse", false);

        // axis is normalized earlier via prefix_scan_op
        const std::size_t axis = v.at("axis").to<std::size_t>();

        std::size_t nslices = 1;
        for(std::size_t i = 0; i < output_shape.lens().size(); ++i)
        {
            if(i != axis)
                nslices *= output_shape.lens()[i];
        }

        constexpr std::size_t block_size = 256;
        options.global                   = nslices * block_size;
        options.local                    = block_size;

        auto src = interpolate_string(prefix_scan_sum_kernel,
                                      {{"axis", std::to_string(axis)},
                                       {"exclusive", exclusive ? "true" : "false"},
                                       {"reverse", reverse ? "true" : "false"}});

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
