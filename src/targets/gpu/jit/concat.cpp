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
#include <migraphx/algorithm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const concat_kernel = R"__migraphx__(
#include <migraphx/kernels/concat.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params}) 
{
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto y, ${concat_params}, auto... xs) {
        concat<${axis}>(${concat_args})(${post}, y, xs...);
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
        auto num_of_concat_inputs = v.get("concat_inputs", inputs.size() - 1);
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.params      = "-Wno-float-equal";
        options.kernel_name = v.get("kernel", "concat_kernel");
        auto axis           = find_fast_axis(options.inputs);
        vectorize vec{};
        if(axis != v.at("axis").to<std::size_t>())
            vec = vectorize::elements(ctx, axis, options.inputs);
        options.set_launch_params(
            v, compute_global_for(ctx, get_concat_elements(options.inputs) / vec.size, 256));
        auto src = interpolate_string(
            concat_kernel,
            {{"kernel", options.kernel_name},
             {"params", enum_params(inputs.size(), "void * private_p")},
             {"args", enum_params(inputs.size(), "private_p")},
             {"concat_params", enum_params(num_of_concat_inputs, "auto concat_x")},
             {"concat_args", enum_params(num_of_concat_inputs, "concat_x")},
             {"post", v.get("post", std::string{"op::id{}"})},
             {"transformers", make_transformer_args(vec)},
             {"preamble", v.get("preamble", std::string{})},
             {"axis", v.at("axis").to<std::string>()}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v = op.to_value();
        if(not ins->module_inputs().empty())
        {
            auto* pm           = ins->module_inputs().front();
            v["concat_inputs"] = ins->inputs().size() - pm->get_parameter_names().size();
            v["preamble"]      = generate_pointwise(*pm, "post_concat");
            v["post"]          = "MIGRAPHX_LIFT(post_concat)";
            v["kernel"]        = "concat_" + generate_name_from_ops(*pm) + "_kernel";
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

// NOLINTNEXTLINE
static const char* const fused_concat_kernel = R"__migraphx__(
#include <migraphx/kernels/concat.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/ops.hpp>
#include <args.hpp>

namespace migraphx {

${preamble}

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params}) 
{
    transform_args(make_tensors(), rotate_last(), ${transformers})(${args})([](auto y, ${concat_params}, auto... xs) {
        concat2<${axis}>(${concat_args})(${post}, y, xs...);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct fused_concat_compiler : compiler<fused_concat_compiler>
{
    std::vector<std::string> names() const { return {"fused_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.params      = "-Wno-float-equal";
        options.kernel_name = v.get("kernel", "concat_kernel");
        auto axis           = find_fast_axis(options.inputs);
        auto op_names       = v.at("ops").to_vector<std::string>();
        auto args           = v.at("args");
        vectorize vec{};
        if(axis != v.at("axis").to<std::size_t>())
            vec = vectorize::elements(ctx, axis, options.inputs);
        auto nelements_per_op = options.inputs.back().elements() / op_names.size();
        options.set_launch_params(v, compute_global_for(ctx, nelements_per_op / vec.size, 256));
        std::vector<std::string> concat_params;
        std::vector<std::string> concat_args;
        for(const auto& name : op_names)
        {
            auto n      = args.at(name).to<std::size_t>();
            auto prefix = name + "_concat_x";
            transform(range(n), std::back_inserter(concat_params), [&](auto i) {
                return "auto " + prefix + std::to_string(i);
            });
            std::vector<std::string> pack_args = {"MIGRAPHX_LIFT(" + name + ")"};
            transform(range(n), std::back_inserter(pack_args), [&](auto i) {
                return prefix + std::to_string(i);
            });
            concat_args.push_back("pack(" + join_strings(pack_args, ", ") + ")");
        }
        auto src = interpolate_string(fused_concat_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"concat_params", join_strings(concat_params, ", ")},
                                       {"concat_args", join_strings(concat_args, ", ")},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"transformers", make_transformer_args(vec)},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"axis", v.at("axis").to<std::string>()}});
        return compile_hip_code_object(src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v = op.to_value();
        std::unordered_map<std::string, std::string> mod_names_lookup;
        transform(range(ins->module_inputs().size()),
                  std::inserter(mod_names_lookup, mod_names_lookup.end()),
                  [&](auto i) {
                      return std::make_pair(ins->module_inputs()[i]->name(),
                                            "pointwise" + std::to_string(i));
                  });
        v["preamble"] = transform_accumulate(
            ins->module_inputs().begin(),
            ins->module_inputs().end(),
            std::string{},
            std::plus<>{},
            [&](module_ref mod) {
                return generate_pointwise(*mod, mod_names_lookup.at(mod->name())) + "\n";
            });
        std::vector<std::string> mod_names;
        std::transform(ins->module_inputs().begin(),
                       ins->module_inputs().end(),
                       std::back_inserter(mod_names),
                       [&](module_ref mod) { return mod_names_lookup.at(mod->name()); });
        v["ops"] = mod_names;
        std::unordered_map<std::string, std::size_t> mod_args;
        std::transform(ins->module_inputs().begin(),
                       ins->module_inputs().end(),
                       std::inserter(mod_args, mod_args.end()),
                       [&](module_ref mod) {
                           const auto& name = mod_names_lookup.at(mod->name());
                           return std::make_pair(name, mod->get_parameter_names().size());
                       });
        v["args"]   = mod_args;
        v["kernel"] = transform_accumulate(
                          ins->module_inputs().begin(),
                          ins->module_inputs().end() - 1,
                          std::string{},
                          std::plus<>{},
                          [&](module_ref mod) { return generate_name_from_ops(*mod) + "_"; }) +
                      "concat_" + generate_name_from_ops(*(ins->module_inputs().back())) +
                      "_kernel";
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
