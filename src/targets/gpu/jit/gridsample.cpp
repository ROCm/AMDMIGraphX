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
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/op/gridsample.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const gridsample_kernel = R"__migraphx__(
#include <migraphx/kernels/gridsample.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void gridsample_kernel(void* in_x, void* in_grid, void* y)
{
    make_tensors()(in_x, in_grid, y)([](auto&&... xs) {
        gridsample<bool{ALIGN_CORNERS},
                   gridsample_padding::PADDING_MODE,
                   gridsample_mode::GRID_MODE>(xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct gridsample_compiler : compiler<gridsample_compiler>
{
    std::vector<std::string> names() const { return {"gridsample"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements()), 256);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "gridsample_kernel";

        options.emplace_param("-DALIGN_CORNERS=" +
                              std::string(v.at("align_corners").to<bool>() ? "true" : "false"));

        // enum gridsample_padding in kernels/gridsample.hpp repeats these
        // enumerator names, so the name is what gets pasted onto its scope.
        options.emplace_param("-DPADDING_MODE=" +
                              to_string(v.at("padding_mode").to<op::gridsample::padding>()));

        // The opset-16 spellings select the same kernel as their opset-20 names,
        // so they are folded here; enum gridsample_mode carries only the three
        // opset-20 names.
        op::gridsample::sample_mode mode = v.at("mode").to<op::gridsample::sample_mode>();
        if(mode == op::gridsample::sample_mode::bilinear)
            mode = op::gridsample::sample_mode::linear;
        else if(mode == op::gridsample::sample_mode::bicubic)
            mode = op::gridsample::sample_mode::cubic;

        options.emplace_param("-DGRID_MODE=" + to_string(mode));

        return compile_hip_code_object(ctx, gridsample_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
