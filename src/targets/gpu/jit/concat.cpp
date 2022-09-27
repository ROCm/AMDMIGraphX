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
#include <migraphx/reduce_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const concat_kernel = R"__migraphx__(
#include <migraphx/kernels/concat.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

__global__ void ${kernel}(${params}) 
{
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto y, auto... xs) {
        concat<${axis}>(y, xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct concat_compiler : compiler<concat_compiler>
{
    std::vector<std::string> names() const { return {"concat"}; }

    static std::size_t get_concat_elements(const std::vector<shape>& inputs)
    {
        return inputs.back().elements() / (inputs.size() - 1);
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // TODO: Use reduce_dims
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.params      = "-Wno-float-equal";
        auto axis           = find_fast_axis(options.inputs);
        auto vec            = vectorize::elements(ctx, axis, options.inputs);
        options.kernel_name = v.get("kernel", "concat_kernel");
        options.set_launch_params(
            v, compute_global_for(ctx, get_concat_elements(options.inputs) / vec.size, 256));
        auto src = interpolate_string(concat_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"transformers", make_transformer_args(vec)},
                                       {"axis", v.at("axis").to<std::string>()}});
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
