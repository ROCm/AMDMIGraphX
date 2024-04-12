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
        if(gemm_ins != smod->end() and std::distance(gemm_ins, smod->end()) > 2)
        {
            auto input_args = ins->inputs();
            input_args.pop_back();
            auto mod2                = smod->split(input_args, {gemm_ins});
            auto dot_mlir_inputs = to_shapes(mod2[0].inputs);
            dot_mlir_inputs.push_back(mod2[0].mod.get_output_shapes().front());
            auto cop1        = compile_mlir(ctx, mod2[0].mod, dot_mlir_inputs, solution);
            auto pw_inputs   = mod2[1].inputs;
            auto dot_ins_idx = std::distance(
                std::find(pw_inputs.begin(), pw_inputs.end(), gemm_ins), pw_inputs.begin());
            auto pw_shapes = to_shapes(mod2[1].inputs);
            pw_shapes[dot_ins_idx] = cop1.output;
            pw_shapes.push_back(mod2[1].mod.get_output_shapes().front());
            assert(pw_shapes.back() == ins->get_shape());
            auto cop2 = compile_pointwise(ctx, pw_shapes, &mod2[1].mod);
            std::vector<operation> cops = {cop1, cop2};
            return insert(cops, mod2, gemm_ins);
        }
        return insert(compile_mlir(ctx, *smod, to_shapes(ins->inputs()), solution));
    }

    compiler_replace insert(code_object_op cobj) const
    {
        return {std::vector<operation>{std::move(cobj)},
                [](module& m,
                   instruction_ref ins,
                   const std::vector<operation>& ops,
                   const std::unordered_map<instruction_ref, instruction_ref>&) {
                    auto mlir =
                        insert_mlir(m, ins, any_cast<code_object_op>(ops.front()), ins->inputs());
                    return m.replace_instruction(ins, mlir);
                }};
    }

    compiler_replace insert(std::vector<operation> cobjs, std::array<module_with_inputs, 2> mods, instruction_ref split_ins) const
    {
        return {std::move(cobjs),
                [=](module& m,
                    instruction_ref ins,
                    const std::vector<operation>& ops,
                    const std::unordered_map<instruction_ref, instruction_ref>& inputs_rep_map) {
                    auto dot_inputs = mods[0].inputs;
                    auto dot_mod_out_shape = mods[0].mod.get_output_shapes().front();
                    if(dot_mod_out_shape == ins->inputs().back()->get_shape())
                    {
                        dot_inputs.push_back(ins->inputs().back());
                    }
                    else
                    {
                        auto dot_alloc = m.insert_instruction(
                            ins,
                            migraphx::make_op("hip::allocate",
                                              {{"shape", to_value(dot_mod_out_shape)}}));
                        dot_inputs.push_back(dot_alloc);
                    }
                    std::vector<instruction_ref> dot_inputs_updated;
                    std::transform(dot_inputs.begin(),
                                   dot_inputs.end(),
                                   std::back_inserter(dot_inputs_updated),
                                   [&](const auto& i) {
                                       if(inputs_rep_map.find(i) != inputs_rep_map.end())
                                       {
                                           assert(inputs_rep_map.at(i)->get_shape() ==
                                                  i->get_shape());
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
                                           assert(inputs_rep_map.at(i)->get_shape() ==
                                                  i->get_shape());
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
