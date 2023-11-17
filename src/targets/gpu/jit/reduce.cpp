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
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void reduce_kernel(void* input_p, void* output_p) 
{
    
    transform_args(make_tensors(), ${transformers})(input_p, output_p)([](auto input, auto output) {

        simple_reduce<reduce::${algo}>(${reduction}, ${init}, input, output, ${read}, ${write});
    });
}
    
}

} // namespace migraphx

)__migraphx__";

static std::vector<std::size_t> get_reduce_lens(const std::vector<std::size_t>& input_lens,
                                                const std::vector<std::size_t>& output_lens)
{
    std::vector<std::size_t> reduce_lens;
    std::transform(output_lens.begin(),
                   output_lens.end(),
                   input_lens.begin(),
                   std::back_inserter(reduce_lens),
                   [](auto x, auto y) -> std::size_t {
                       if(x == y)
                           return 1;
                       else
                           return y;
                   });
    return reduce_lens;
}

template <class T>
static shape get_reduced_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    std::fill(lens.begin(), lens.end(), 1);
    for(const auto& axis : axes)
        lens[axis] = s.lens()[axis];
    return s.with_lens(lens);
}

template <class T>
static shape get_output_shape(const shape& s, const std::vector<T>& axes)
{
    auto lens = s.lens();
    for(const auto& axis : axes)
        lens[axis] = 1;
    return s.with_lens(lens);
}

template <class ReduceLens>
static std::string get_reduce_algo(const std::vector<shape>& inputs, ReduceLens rlens)
{
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
    return "block";
}

static std::string get_reduce_algo(const std::vector<shape>& inputs)
{
    auto rlens = get_reduce_lens(inputs.front().lens(), inputs.back().lens());
    return get_reduce_algo(inputs, rlens);
}

struct simple_reduce_compiler : compiler<simple_reduce_compiler>
{
    std::vector<std::string> names() const
    {
        return {"simple_reduce",
                "reduce_sum",
                "reduce_mean",
                "reduce_max",
                "reduce_min",
                "reduce_prod"};
    }

    static std::size_t get_reduce_elements(const std::vector<shape>& inputs)
    {
        return inputs.front().elements() / inputs.back().elements();
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = reduce_dims(inputs);
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        auto nelements = options.virtual_inputs.back().elements();
        auto algo      = v.get("algo", get_reduce_algo(options.virtual_inputs));

        if(algo == "block")
        {
            // Vectorize if the axis is a reduction axis
            if(options.virtual_inputs.back().lens()[faxis] == 1)
                vec = vectorize::elements(ctx, faxis, options.virtual_inputs);
            auto relements  = get_reduce_elements(options.virtual_inputs) / vec.size;
            auto block_size = compute_block_size(relements, 256);
            if(relements >= block_size * 256)
                algo = "block_large";
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
        options.kernel_name  = "reduce_kernel";
        std::string identity = "[](auto x) { return x; }";
        auto src             = interpolate_string(simple_reduce_kernel,
                                                  {{"reduction", v.at("reduction").to<std::string>()},
                                                   {"init", v.get("init", std::string{"0"})},
                                                   {"read", v.get("read", identity)},
                                                   {"write", v.get("write", identity)},
                                                   {"algo", algo},
                                                   {"transformers", make_transformer_args(vec)},
                                                   {"preamble", v.get("preamble", std::string{})}});
        // disable DPP for FP8 for now,, TODO: need to disable for Any FP8 types
        if(std::any_of(inputs.begin(), inputs.end(), [](const auto& s) {
               return s.type() == migraphx::shape::fp8e4m3fnuz_type;
           }))
        {
            options.params += "-DMIGRAPHX_HAS_DPP=0 ";
        }
        options.params += "-Wno-float-equal";
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        value v = value::object{};
        reduce_op r{};
        r.set(ins, op);
        v["reduction"] = r.reduction;
        v["read"]      = r.read;
        v["write"]     = r.write;
        v["init"]      = r.init;
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

static const char* const fused_reduce_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/reduce.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {
MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto y, auto... xs) {
        fused_reduce<reduce::${algo}, ${reduced}>(y, partial(${lambda})(xs...));
    });
}
    
}

} // namespace migraphx

)__migraphx__";

struct fused_reduce_compiler : compiler<fused_reduce_compiler>
{
    std::vector<std::string> names() const { return {"fused_reduce"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto axes           = v.at("axes").to_vector<std::size_t>();
        auto virtual_inputs = inputs;
        virtual_inputs.push_back(get_reduced_shape(inputs.front(), axes));
        virtual_inputs.push_back(get_output_shape(inputs.front(), axes));
        virtual_inputs           = reduce_dims(normalize_permutation(virtual_inputs));
        auto reduce_output_shape = virtual_inputs.back();
        virtual_inputs.pop_back();
        auto reduction_shape = virtual_inputs.back();
        virtual_inputs.pop_back();

        hip_compile_options options;
        options.inputs         = inputs;
        options.output         = inputs.back();
        options.virtual_inputs = virtual_inputs;
        auto faxis             = find_fast_axis({options.virtual_inputs.front()});
        vectorize vec{};
        auto nelements = reduce_output_shape.elements();
        auto algo = v.get("algo", get_reduce_algo(options.virtual_inputs, reduction_shape.lens()));
        if(algo == "block")
        {
            // Vectorize if the axis is a reduction axis
            if(reduce_output_shape.lens()[faxis] == 1)
                vec = vectorize::elements(ctx, faxis, options.virtual_inputs);
            auto relements  = reduction_shape.elements() / vec.size;
            auto block_size = compute_block_size(relements, 256);
            if(relements >= block_size * 256)
                algo = "block_large";
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
        options.kernel_name = v.get("kernel", "reduce_kernel");
        auto src            = interpolate_string(
            fused_reduce_kernel,
            {{"kernel", options.kernel_name},
                        {"params", enum_params(inputs.size(), "void * private_p")},
                        {"args", enum_params(inputs.size(), "private_p")},
                        {"algo", algo},
                        {"reduced", "decltype(" + generate_make_shape(reduce_output_shape) + ")"},
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
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
