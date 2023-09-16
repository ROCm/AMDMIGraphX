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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const scatter_elements_kernel = R"__migraphx__(
#include <migraphx/kernels/scatter_elements.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void scatter_elements_kernel(void* in_indices, void* in_updates, void* output) 
{
    make_tensors()(in_indices, in_updates, output)([](auto&&... xs) { 
        scatter_elements<${axis}>(xs..., ${reduction}{}); 
    });
}

}

} // namespace migraphx

)__migraphx__";

struct scatter_elements_compiler : compiler<scatter_elements_compiler>
{
    std::vector<std::string> names() const { return {"scatter_elements"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.at(1).elements()));
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.kernel_name    = "scatter_elements_kernel";
        options.virtual_inputs = inputs;
        auto reduction         = "assign_" + v.get("reduction", std::string{"none"});
        auto axis              = std::to_string(v.get("axis", 0));
        auto src =
            interpolate_string(scatter_elements_kernel, {{"reduction", reduction}, {"axis", axis}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return insert(compile_op(
            ctx,
            to_shapes(std::vector<instruction_ref>{ins->inputs().begin() + 1, ins->inputs().end()}),
            op.to_value()));
    }

    compiler_replace insert(const operation& co) const
    {
        return {co, [](module& m, instruction_ref ins, const operation& op) {
                    auto args = ins->inputs();
                    args.back() =
                        m.insert_instruction(ins, make_op("hip::copy"), args.front(), args.back());
                    args.erase(args.begin());
                    return m.replace_instruction(ins, op, args);
                }};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
