/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#if !MIGRAPHX_USE_MIOPEN
#include <migraphx/op/pooling.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const topk_kernel = R"__migraphx__(
#include <migraphx/kernels/topk.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void topk_kernel(void* in_x, void* out_y, void* out_indices) 
{
    make_tensors()(in_x, out_y, out_indices)([](auto x, auto y, auto indices) {
        topk<${axis}>(y, indices, x, ${compare}, ${init});
    });
}

}

} // namespace migraphx

)__migraphx__";

struct topk_compiler : compiler<topk_compiler>
{
    std::vector<std::string> names() const { return {"topk"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.output      = inputs.back();
        options.inputs      = flatten(inputs);
        options.kernel_name = "topk_kernel";


        auto axis = v.at("axis").to<int64_t>();
        auto relements = inputs.front().lens()[axis];
        auto nelements = inputs.front().elements() / relements;
        auto block_size = compute_block_size(ctx, relements/2, 256);
        options.set_launch_params(v, compute_global_for(ctx, block_size*nelements), block_size);

        std::string compare = "less{}";
        std::string init = "highest{}";

        if(v.at("largest").to<bool>())
        {
            compare = "greater{}";
            init = "lowest{}";
        }

        auto src = interpolate_string(
            topk_kernel,
            {
             {"compare", compare},
             {"init", init},
             {"axis", std::to_string(axis)}
         });

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
