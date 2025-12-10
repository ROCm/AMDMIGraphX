/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iterator>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_DUMP_TO_MXR);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLIR_DUMP);

static module create_pointwise_module(module_ref in_mod)
{
    module pw_mod;
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    for(auto param : in_mod->get_parameters())
    {
        map_ins[param] =
            pw_mod.add_parameter(any_cast<builtin::param>(param->get_operator()).parameter,
                                 shape{param->get_shape().type()});
    }
    auto return_args =
        pw_mod.add_instructions(in_mod,
                                &map_ins,
                                [](module& m,
                                   instruction_ref ins,
                                   const operation& op,
                                   const std::vector<instruction_ref>& inputs,
                                   const std::vector<module_ref>& mod_args) -> instruction_ref {
                                    auto out_alias = op.output_alias(to_shapes(inputs));
                                    if(out_alias >= 0)
                                        return inputs.at(out_alias);
                                    else
                                        return m.insert_instruction(ins, op, inputs, mod_args);
                                });
    pw_mod.add_return(return_args);
    return pw_mod;
}

static code_object_op
compile_pointwise_module(context& ctx, const std::vector<shape>& inputs, module_ref mod)
{
    operation cop;
    auto pw_mod = create_pointwise_module(mod);
    if(any_of(mod->get_parameters(), [&](instruction_ref param) {
           if(param->outputs().size() != 1)
               return false;
           return instruction::get_output_alias(param->outputs().front(), /* shallow */ true) ==
                  param;
       }))
    {
        auto mod2 = *mod;
        adjust_param_shapes(mod2, inputs);
        auto names = mod2.get_parameter_names();
        std::sort(names.begin(), names.end());
        std::vector<shape> new_shapes;
        std::transform(names.begin(),
                       names.end(),
                       std::back_inserter(new_shapes),
                       [&](const std::string& name) {
                           auto param       = mod2.get_parameter(name);
                           auto output_path = get_output_path(param);
                           auto it          = std::adjacent_find(
                               output_path.begin(),
                               output_path.end(),
                               [&](instruction_ref, instruction_ref output) {
                                   return instruction::get_output_alias(output) != param;
                               });
                           return (*it)->get_shape();
                       });
        std::copy(inputs.begin() + new_shapes.size(), inputs.end(), std::back_inserter(new_shapes));
        cop = compile_pointwise(ctx, new_shapes, &pw_mod);
    }
    else
    {
        cop = compile_pointwise(ctx, inputs, &pw_mod);
    }
    auto co            = any_cast<code_object_op>(cop);
    co.expected_inputs = inputs;
    return co;
}

static instruction_ref find_final_split(instruction_ref split_ins)
{
    auto output_path = get_output_path(split_ins);
    auto it          = std::adjacent_find(
        output_path.begin(), output_path.end(), [&](instruction_ref input, instruction_ref output) {
            if(contains({"reshape", "squeeze", "unsqueeze", "transpose"}, output->name()))
                return false;
            if(contains({"add", "mul"}, output->name()))
            {
                auto aux = *std::find_if(output->inputs().begin(),
                                         output->inputs().end(),
                                         [&](instruction_ref i) { return i != input; });
                if(aux->can_eval())
                    return false;
                return instruction::get_output_alias(aux)->name() != "@param";
            }
            return true;
        });
    return *it;
}

struct mlir_compiler : compiler<mlir_compiler>
{
    std::vector<std::string> names() const { return {"gpu::mlir_op"}; }

    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }
    std::optional<instruction_ref> input_is_param(const instruction_ref& ins) const
    {
        auto cur = instruction::get_output_alias(ins);
        while(contains({"reshape", "contiguous"}, cur->name()))
        {
            cur = instruction::get_output_alias(cur->inputs().at(0));
        }
        if(cur->name() == "@param")
        {
            return cur;
        }
        return nullopt;
    }

    bool is_range_literal(const instruction_ref& ins) const
    {
        auto lit_elms = ins->get_shape().element_space();
        if(not ins->can_eval() or lit_elms < 2)
        {
            return false;
        }
        bool is_range = false;
        ins->eval().visit([&](auto l) {
            is_range = std::adjacent_find(l.begin(), l.begin() + lit_elms, [](auto cur, auto next) {
                           return not float_equal(next - cur, 1.0);
                       }) == l.begin() + lit_elms;
        });
        return is_range;
    }

    void set_fill_map(compiler_replace& cr, const module& m) const
    {
        for(auto ins : iterator_for(m))
        {
            if(ins->name() != "greater")
            {
                continue;
            }
            auto fill_val      = ins->get_shape().lens().back() - 1;
            bool has_range_lit = std::any_of(ins->inputs().begin(),
                                             ins->inputs().end(),
                                             [&](auto inp) { return is_range_literal(inp); });
            for(auto inp : ins->inputs())
            {
                auto param = input_is_param(inp);
                if(param.has_value() and has_range_lit)
                {
                    auto id = param.value()->get_shape().type_string() +
                              migraphx::shape::to_sizes_string({param.value()->get_shape()});
                    cr.fill_map[id] = static_cast<double>(fill_val);
                }
            }
        }
    }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation&, const value& solution) const
    {
        auto* smod = ins->module_inputs().front();
        assert(smod->get_parameter_names().size() == ins->inputs().size() - 1);
        auto gemm_like_ins = std::find_if(smod->begin(), smod->end(), [&](const auto& i) {
            return contains({"dot", "quant_dot", "convolution", "quant_convolution"}, i.name());
        });
        auto pointwise_ins = std::find_if(gemm_like_ins, smod->end(), [&](const auto& i) {
            return i.get_operator().attributes().get("pointwise", false) == true;
        });

        // check if (a) module is fused (b) contains a "gemm/conv" instruction and (c)
        // perfConfig can not allow fused module
        if(gemm_like_ins != smod->end() and pointwise_ins != smod->end() and
           not is_module_fusible(*smod, ctx, solution))
        {
            auto input_args = ins->inputs();
            // remove alloc buffer
            input_args.pop_back();
            auto split_ins                               = find_final_split(gemm_like_ins);
            std::array<module_with_inputs, 2> mod_splits = smod->split(input_args, {split_ins});
            auto dot_mlir_inputs = to_shapes(mod_splits[0].inputs);
            // add alloc for the gemm output
            dot_mlir_inputs.push_back(mod_splits[0].mod.get_output_shapes().front());
            mlir_code_object cop1 = compile_mlir(ctx, mod_splits[0].mod, dot_mlir_inputs, solution);
            auto pw_shapes        = to_shapes(mod_splits[1].inputs);
            if(mod_splits[1].mod.get_output_shapes().size() == 1)
            {
                pw_shapes.push_back(mod_splits[1].mod.get_output_shapes().front());
            }
            else
            {
                pw_shapes.push_back(shape{mod_splits[1].mod.get_output_shapes()});
            }
            assert(pw_shapes.back() == ins->get_shape());
            auto cop2 = compile_pointwise_module(ctx, pw_shapes, &mod_splits[1].mod);
            std::vector<mlir_code_object> cops = {cop1, mlir_code_object{cop2}};
            return insert(cops, mod_splits, ins, split_ins);
        }
        auto cr = insert(compile_mlir(ctx, *smod, to_shapes(ins->inputs()), solution));
        set_fill_map(cr, *smod);
        return cr;
    }

    compiler_replace insert(const mlir_code_object& mco) const
    {
        return {std::vector<operation>{mco.cop},
                [=](module& m, instruction_ref ins, const std::vector<operation>& ops) {
                    std::vector<instruction_ref> inputs = ins->inputs();

                    // Tuple inputs not supported
                    assert(std::all_of(inputs.begin(), inputs.end() - 1, [](auto i) {
                        return i->get_shape().sub_shapes().empty();
                    }));

                    // Multiple output case (allocate ins will give a tuple)
                    std::vector<instruction_ref> flat_inputs(inputs);
                    bool multi_out = not flat_inputs.back()->get_shape().sub_shapes().empty();
                    if(multi_out)
                    {
                        auto allocs = flat_inputs.back();
                        flat_inputs.pop_back();
                        auto sub_shape_idx = range(allocs->get_shape().sub_shapes().size());
                        std::transform(sub_shape_idx.begin(),
                                       sub_shape_idx.end(),
                                       std::back_inserter(flat_inputs),
                                       [&](int i) {
                                           return m.insert_instruction(
                                               ins,
                                               migraphx::make_op("get_tuple_elem", {{"index", i}}),
                                               allocs);
                                       });
                    }
                    std::vector<instruction_ref> tuple_replacements;

                    for(const auto i : range(mco.prefill_indices.size()))
                    {
                        auto prefilled_ins = m.insert_instruction(
                            ins,
                            migraphx::make_op("hip::fill", {{"value", mco.prefill_values[i]}}),
                            flat_inputs[mco.prefill_indices[i]]);
                        if(not multi_out or mco.prefill_indices[i] < inputs.size() - 1)
                        {
                            replace(inputs, inputs[mco.prefill_indices[i]], prefilled_ins);
                        }
                        else
                        {
                            tuple_replacements.push_back(prefilled_ins);
                        }
                    }

                    if(multi_out and not tuple_replacements.empty())
                    {
                        // Add identity to make sure fill operations happen before kernel call
                        tuple_replacements.insert(tuple_replacements.begin(), inputs.back());
                        inputs.back() = m.insert_instruction(
                            ins, migraphx::make_op("identity"), tuple_replacements);
                    }

                    auto mlir = insert_mlir(m, ins, any_cast<code_object_op>(ops.front()), inputs);
                    return m.replace_instruction(ins, mlir);
                },
                &trace};
    }

    compiler_replace insert(const std::vector<mlir_code_object>& mcos,
                            const std::array<module_with_inputs, 2>& mods,
                            instruction_ref precompile_ins,
                            instruction_ref split_ins) const
    {
        std::vector<operation> cobjs(mcos.size());
        std::transform(
            mcos.begin(), mcos.end(), cobjs.begin(), [](const auto& mco) { return mco.cop; });
        auto precompiled_inputs = precompile_ins->inputs();
        return {
            cobjs, [=](module& m, instruction_ref ins, const std::vector<operation>& ops) {
                auto compiled_inputs = ins->inputs();
                std::unordered_map<instruction_ref, instruction_ref> inputs_rep_map;
                for(const auto i : range(precompiled_inputs.size()))
                {
                    inputs_rep_map[precompiled_inputs[i]] = compiled_inputs[i];
                }
                auto dot_inputs        = mods[0].inputs;
                auto dot_mod_out_shape = mods[0].mod.get_output_shapes().front();
                auto dot_alloc         = m.insert_instruction(
                    ins,
                    migraphx::make_op("hip::allocate", {{"shape", to_value(dot_mod_out_shape)}}));
                dot_inputs.push_back(dot_alloc);
                for(const auto i : range(mcos[0].prefill_indices.size()))
                {
                    auto prefilled_ins = m.insert_instruction(
                        ins,
                        migraphx::make_op("hip::fill", {{"value", mcos[0].prefill_values[i]}}),
                        dot_inputs[mcos[0].prefill_indices[i]]);
                    replace(dot_inputs, dot_inputs[mcos[0].prefill_indices[i]], prefilled_ins);
                }

                std::vector<instruction_ref> dot_inputs_updated;
                std::transform(dot_inputs.begin(),
                               dot_inputs.end(),
                               std::back_inserter(dot_inputs_updated),
                               [&](const auto& i) {
                                   if(inputs_rep_map.find(i) != inputs_rep_map.end())
                                   {
                                       assert(inputs_rep_map.at(i)->get_shape() == i->get_shape());
                                       return inputs_rep_map.at(i);
                                   }
                                   return i;
                               });
                auto mlir_ins =
                    insert_mlir(m, ins, any_cast<code_object_op>(ops[0]), dot_inputs_updated);
                auto pwm = mods[1];
                pwm.replace(split_ins, mlir_ins);
                auto pw_inputs = pwm.inputs;
                pw_inputs.push_back(ins->inputs().back());
                std::vector<instruction_ref> pw_inputs_updated;
                std::transform(pw_inputs.begin(),
                               pw_inputs.end(),
                               std::back_inserter(pw_inputs_updated),
                               [&](const auto& i) {
                                   if(inputs_rep_map.find(i) != inputs_rep_map.end())
                                   {
                                       assert(inputs_rep_map.at(i)->get_shape() == i->get_shape());
                                       return inputs_rep_map.at(i);
                                   }
                                   return i;
                               });
                auto pw_ins =
                    insert_mlir(m, ins, any_cast<code_object_op>(ops[1]), pw_inputs_updated);
                return m.replace_instruction(ins, pw_ins);
            }};
    }

    optional<tuning_config> get_tuning_config(const context& ctx,
                                              instruction_ref ins,
                                              const operation&,
                                              bool exhaustive) const
    {
        static const auto mxr_loc  = string_value_of(MIGRAPHX_MLIR_DUMP_TO_MXR{});
        static const auto mlir_loc = string_value_of(MIGRAPHX_MLIR_DUMP{});

        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        if(not mxr_loc.empty())
        {
            dump_mlir_to_mxr(*smod, ins->inputs(), mxr_loc);
        }
        if(not mlir_loc.empty())
        {
            dump_mlir_to_file(*smod, shapes, mlir_loc);
        }
        return get_tuning_config_mlir(ctx, *smod, shapes, exhaustive);
    }

    static void trace(std::ostream& os, instruction_ref ins)
    {
        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        os << dump_mlir(*smod, shapes);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
