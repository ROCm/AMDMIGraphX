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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/serialize.hpp>

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
    std::vector<std::string> names() const { return {"fused_concat", "concat"}; }

    static std::vector<shape> normalize(std::vector<shape> inputs, std::size_t& axis)
    {
        if(std::any_of(inputs.begin(), inputs.end(), [](const shape& x) { return x.dynamic(); }))
            return inputs;
        auto s = inputs.back();
        std::vector<std::size_t> strides(s.lens().size());
        strides[axis] = 1;

        inputs.push_back(shape{s.type(), s.lens(), strides});

        auto result   = reduce_dims(normalize_permutation(inputs));
        auto rstrides = result.back().strides();
        auto it = std::find_if(rstrides.begin(), rstrides.end(), [](auto x) { return x == 1; });
        axis    = it - rstrides.begin();
        result.pop_back();
        return result;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const std::size_t kernel_axis = v.at("axis").to<std::size_t>();
        const std::size_t num_concat  = v.get("num_concat_inputs", inputs.size());
        std::vector<shape> concat_shapes;
        concat_shapes.assign(inputs.begin(), inputs.begin() + std::min(num_concat, inputs.size()));
        shape output_shape =
            v.contains("output_shape") ? from_value<shape>(v.at("output_shape")) : inputs.back();

        options.inputs      = inputs;
        options.output      = output_shape;
        options.kernel_name = v.get("kernel", "concat_kernel");

        // normalize() rewrites axis into reduced-dim space; kernel concat<Axis> uses full tensors.
        std::size_t fast_axis          = kernel_axis;
        std::vector<shape> norm_concat = normalize(concat_shapes, fast_axis);
        const bool any_dynamic         = std::any_of(
            concat_shapes.begin(), concat_shapes.end(), [](const shape& x) { return x.dynamic(); });
        auto axis = any_dynamic ? kernel_axis : find_fast_axis(norm_concat);
        // virtual_inputs must match inputs for compile_hip_code_object unless empty
        options.virtual_inputs = {};

        auto op_names = v.at("ops").to_vector<std::string>();
        auto args     = v.at("args");
        // Output-alias path (precompile_op): operand shapes differ from the output buffer on
        // the concat axis, so vectorization must be disabled to avoid half2/scalar mismatches.
        const bool has_output_alias = num_concat < inputs.size();
        vectorize vec{};
        if(not any_dynamic and not has_output_alias and axis != kernel_axis)
            vec = vectorize::elements(ctx, axis, norm_concat);

        const std::size_t nelem =
            output_shape.dynamic() ? output_shape.element_space() : output_shape.elements();
        auto nelements_per_op = nelem / op_names.size();
        options.set_launch_params(v, compute_global_for(ctx, nelements_per_op / vec.size, 256));
        options.emplace_param("-Wno-float-equal");
        std::vector<std::string> concat_params;
        std::vector<std::string> concat_args;
        for(auto i : range(op_names.size()))
        {
            const auto& name = op_names[i];
            auto n           = args.at(name).to<std::size_t>();
            auto prefix      = to_c_id(name + std::to_string(i) + "_concat_x");
            transform(range(n), std::back_inserter(concat_params), [&](auto j) {
                return "auto " + prefix + std::to_string(j);
            });
            std::vector<std::string> pack_args = {"MIGRAPHX_LIFT(" + name + ")"};
            transform(range(n), std::back_inserter(pack_args), [&](auto j) {
                return prefix + std::to_string(j);
            });
            concat_args.push_back("pack(" + join_strings(pack_args, ", ") + ")");
        }
        auto src = interpolate_string(concat_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void * private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"concat_params", join_strings(concat_params, ", ")},
                                       {"concat_args", join_strings(concat_args, ", ")},
                                       {"post", v.get("post", std::string{"op::id{}"})},
                                       {"transformers", make_transformer_args(vec)},
                                       {"preamble", v.get("preamble", std::string{})},
                                       {"axis", std::to_string(kernel_axis)}});
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v = op.to_value();
        if(op.name() == "fused_concat")
        {
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
                           ins->module_inputs().end() - 1,
                           std::back_inserter(mod_names),
                           [&](module_ref mod) { return mod_names_lookup.at(mod->name()); });
            v["ops"]            = mod_names;
            module_ref last_mod = ins->module_inputs().back();
            v["post"]           = "MIGRAPHX_LIFT(" + mod_names_lookup.at(last_mod->name()) + ")";
            std::unordered_map<std::string, std::size_t> mod_args;
            std::transform(ins->module_inputs().begin(),
                           ins->module_inputs().end() - 1,
                           std::inserter(mod_args, mod_args.end()),
                           [&](module_ref mod) {
                               const auto& name = mod_names_lookup.at(mod->name());
                               return std::make_pair(name, mod->get_parameter_names().size());
                           });
            v["args"]        = mod_args;
            auto prefix_name = transform_accumulate(ins->module_inputs().begin(),
                                                    ins->module_inputs().end() - 1,
                                                    std::string{},
                                                    std::plus<>{},
                                                    [&](module_ref mod) -> std::string {
                                                        auto name = generate_name_from_ops(*mod);
                                                        if(name.empty())
                                                            return "";
                                                        return name + "_";
                                                    });
            v["kernel"]      = prefix_name + "concat_" +
                          generate_name_from_ops(*(ins->module_inputs().back())) + "_kernel";
        }
        else if(op.name() == "concat")
        {
            std::size_t concat_inputs = ins->inputs().size();
            if(ins->name() == "gpu::precompile_op")
                concat_inputs -= 1;
            if(not ins->module_inputs().empty())
            {
                auto* pm      = ins->module_inputs().front();
                concat_inputs = ins->inputs().size() - pm->get_parameter_names().size();
                v["preamble"] = generate_pointwise(*pm, "post_concat");
                v["post"]     = "MIGRAPHX_LIFT(post_concat)";
                v["kernel"]   = "concat_" + generate_name_from_ops(*pm) + "_kernel";
            }
            std::vector<std::string> mod_names(concat_inputs, "op::id{}");
            v["ops"]                                              = mod_names;
            std::unordered_map<std::string, std::size_t> mod_args = {{"op::id{}", 1}};
            v["args"]                                             = mod_args;
            v["num_concat_inputs"]                                = concat_inputs;
            v["output_shape"]                                     = to_value(ins->get_shape());
        }
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
