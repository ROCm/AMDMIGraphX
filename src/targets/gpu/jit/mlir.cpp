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

#include <iterator>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <migraphx/gpu/compile_pointwise.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

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
    auto return_args = pw_mod.add_instructions(
        in_mod,
        &map_ins,
        [](module& m,
           instruction_ref ins,
           const operation& op,
           const std::vector<instruction_ref>& inputs,
           const std::vector<module_ref>& mod_args) -> instruction_ref {
            if(op.name() == "multibroadcast" and inputs.front()->name() == "@literal")
                return inputs.front();
            else
                return m.insert_instruction(ins, op, inputs, mod_args);
        });
    pw_mod.add_return(return_args);
    return pw_mod;
}

struct mlir_compiler : compiler<mlir_compiler>
{
    std::vector<std::string> names() const { return {"gpu::mlir_op"}; }

    operation compile_op(context&, const std::vector<shape>&, const value&) const { return {}; }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation&, const value& solution) const
    {
        auto* smod = ins->module_inputs().front();
        assert(smod->get_parameter_names().size() == ins->inputs().size() - 1);
        auto gemm_ins = std::find_if(smod->begin(), smod->end(), [&](const auto& i) {
            return i.name() == "dot" or i.name() == "quant_dot";
        });
        // check if (a) module is fused (b) contains a dot instruction and (c) perfConfig can not
        // allow fused module
        if(gemm_ins != smod->end() and std::distance(gemm_ins, smod->end()) > 2 and
           not is_module_fusible(*smod, ctx, solution))
        {
            auto input_args = ins->inputs();
            input_args.pop_back();
            auto mod_splits      = smod->split(input_args, {gemm_ins});
            auto dot_mlir_inputs = to_shapes(mod_splits[0].inputs);
            dot_mlir_inputs.push_back(mod_splits[0].mod.get_output_shapes().front());
            mlir_code_object cop1 = compile_mlir(ctx, mod_splits[0].mod, dot_mlir_inputs, solution);
            auto pw_inputs        = mod_splits[1].inputs;
            auto dot_ins_idx      = std::distance(
                std::find(pw_inputs.begin(), pw_inputs.end(), gemm_ins), pw_inputs.begin());
            auto pw_shapes         = to_shapes(mod_splits[1].inputs);
            pw_shapes[dot_ins_idx] = cop1.cop.output;
            pw_shapes.push_back(mod_splits[1].mod.get_output_shapes().front());
            assert(pw_shapes.back() == ins->get_shape());
            auto pw_mod                        = create_pointwise_module(&mod_splits[1].mod);
            auto cop2                          = compile_pointwise(ctx, pw_shapes, &pw_mod);
            std::vector<mlir_code_object> cops = {cop1,
                                                  mlir_code_object{any_cast<code_object_op>(cop2)}};
            return insert(cops, mod_splits, ins, gemm_ins);
        }
        return insert(compile_mlir(ctx, *smod, to_shapes(ins->inputs()), solution));
    }

    compiler_replace insert(const mlir_code_object& mco) const
    {
        return {std::vector<operation>{mco.cop},
                [=](module& m, instruction_ref ins, const std::vector<operation>& ops) {
                    std::vector<instruction_ref> inputs = ins->inputs();
                    for(const auto i : range(mco.prefill_indices.size()))
                    {
                        auto prefilled_ins = m.insert_instruction(
                            ins,
                            migraphx::make_op("hip::fill", {{"value", mco.prefill_values[i]}}),
                            inputs[mco.prefill_indices[i]]);
                        replace(inputs, inputs[mco.prefill_indices[i]], prefilled_ins);
                    }
                    auto mlir = insert_mlir(m, ins, any_cast<code_object_op>(ops.front()), inputs);
                    return m.replace_instruction(ins, mlir);
                }};
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
                auto mlir =
                    insert_mlir(m, ins, any_cast<code_object_op>(ops[0]), dot_inputs_updated);
                assert(contains(mods[1].inputs, split_ins));
                auto pwm = mods[1];
                pwm.replace(split_ins, mlir);
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
        auto shapes = to_shapes(ins->inputs());
        auto* smod  = ins->module_inputs().front();
        return get_tuning_config_mlir(ctx, *smod, shapes, exhaustive);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
