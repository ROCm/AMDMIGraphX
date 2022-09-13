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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

static const char* const pointwise_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void pad_kernel(${params}) 
{
    auto offsets = index_ints<${offsets}>{};
    auto idx = make_index();
    pad(idx, offsets, ${transformers})(${args});
}
    
}

} // namespace migraphx

)__migraphx__";

struct pointwise_compiler : compiler<pointwise_compiler>
{
    std::vector<std::string> names() const { return {"pointwise", "contiguous"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        auto axis              = find_fast_axis(options.virtual_inputs);
        auto vec               = vectorize::elements(axis, options.virtual_inputs);
        auto preloads          = preload::broadcasts(axis, options.virtual_inputs);
        options.kernel_name    = "pad_kernel";
        options.set_launch_params(v, compute_global_for(ctx, inputs.at(1).elements()));

        auto src = interpolate_string(pointwise_kernel,
                                      {{"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"lambda", v.at("lambda").to<std::string>()},
                                       {"transformers", make_transformer_args(preloads, vec)},
                                       {"preamble", v.get("preamble", std::string{})}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
