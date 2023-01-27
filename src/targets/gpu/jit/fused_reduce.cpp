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

static const char* const simple_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
__global__ void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto y, auto... xs) {
        fused_reduce<reduce::${algo}, ${reduced}>(y, partial(${lambda})(xs...));
    });
}
    
}

} // namespace migraphx

)__migraphx__";

template <class T>
static shape get_reduced_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    std::fill(lens.begin(), lens.end(), 1);
    for(const auto& axis : axes)
        lens[axis] = s.lens()[axis];
    return shape{s.type(), lens};
}

template <class T>
static shape get_output_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    for(const auto& axis : axes)
        lens[axis] = 1;
    return shape{s.type(), lens};
}

template <class ReduceLens>
static std::string get_reduce_algo(const std::vector<shape>& inputs, ReduceLens rlens)
{
#if 0
    const auto init = std::numeric_limits<std::size_t>::max();
    // The minimum stride
    auto min_stride = std::inner_product(
        rlens.begin(),
        rlens.end(),
        inputs.front().strides().begin(),
        init,
        [](auto x, auto y) { return std::min(x, y); },
        [](auto len, auto stride) { return len == 1 ? init : stride; });
    if(min_stride > 2)
        return "lane";
#endif
    return "block";
}

struct fused_reduce_compiler : compiler<fused_reduce_compiler>
{
    std::vector<std::string> names() const { return {"fused_reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto axes           = v.at("axes").to_vector<std::size_t>();
        auto virtual_inputs = inputs;
        virtual_inputs.push_back(get_reduced_shape(inputs.front(), axes));
        virtual_inputs.push_back(get_output_shape(inputs.front(), axes));
        virtual_inputs    = reduce_dims(virtual_inputs);
        auto output_shape = virtual_inputs.back();
        virtual_inputs.pop_back();
        auto reduced_shape = virtual_inputs.back();
        virtual_inputs.pop_back();

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = virtual_inputs;
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        auto nelements = options.virtual_inputs.back().elements();
        auto algo = v.get("algo", get_reduce_algo(options.virtual_inputs, reduced_shape.lens()));
        if(algo == "block")
        {
            // Vectorize if the axis is a reduction axis
            if(output_shape.lens()[faxis] == 1)
                vec = vectorize::elements(ctx, faxis, options.virtual_inputs);
            auto relements  = reduced_shape.elements() / vec.size;
            auto block_size = compute_block_size(relements, 256);
            options.set_launch_params(
                v, compute_global_for(ctx, nelements * block_size, 256), block_size);
        }
        else if(algo == "lane")
        {
            options.set_launch_params(v, compute_global_for(ctx, nelements, 256));
        }
        else
        {
            MIGRAPHX_THROW("Unknown reduce algo: " + algo);
        }
        options.kernel_name  = v.get("kernel", "reduce_kernel");
        std::string identity = "[](auto x) { return x; }";
        auto src =
            interpolate_string(simple_reduce_kernel,
                               {{"kernel", options.kernel_name},
                                {"params", enum_params(inputs.size(), "void * private_p")},
                                {"args", enum_params(inputs.size(), "private_p")},
                                {"algo", algo},
                                {"reduced", "decltype(" + generate_make_shape(output_shape) + ")"},
                                {"lambda", v.at("lambda").to<std::string>()},
                                {"transformers", make_transformer_args(vec)},
                                {"preamble", v.get("preamble", std::string{})}});
        options.params += "-Wno-float-equal";
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        assert(not ins->module_inputs().empty());
        auto v        = op.to_value();
        auto* rm      = ins->module_inputs().front();
        v["preamble"] = generate_reduce(*rm, "fused_reduce_op");
        v["lambda"]   = "MIGRAPHX_LIFT(fused_reduce_op)";
        v["kernel"]   = generate_name_from_ops(*rm) + "_kernel";
        return replace(compile_op(ctx, to_shapes(ins->inputs()), v));
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
