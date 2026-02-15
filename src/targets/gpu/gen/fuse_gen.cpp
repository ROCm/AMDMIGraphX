/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/gen/fuse_gen.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

namespace {

// Check if an instruction is already handled by another fusion pass
bool is_already_fused(instruction_ref ins)
{
    return starts_with(ins->name(), "gpu::mlir_op") or
           starts_with(ins->name(), "gpu::precompile_op");
}

struct find_gen_pointwise
{
    auto matcher() const
    {
        return match::name("pointwise")(match::used_once());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        // Skip if already handled
        if(is_already_fused(ins))
            return;

        // Skip multi-output pointwise for now
        if(ins->get_shape().type() == shape::tuple_type)
            return;

        auto* pm = ins->module_inputs().front();

        // Create gen module with the same structure as the pointwise submodule
        // but with tensor-shaped parameters matching the actual input shapes.
        std::string module_name = "gen_" + pm->name();
        module_ref gen_mod      = mpm.create_module(module_name);
        gen_mod->set_bypass();

        // Map outer instruction inputs to gen module params
        std::unordered_map<instruction_ref, instruction_ref> outer_map;
        std::size_t param_idx = 0;
        for(auto input : ins->inputs())
        {
            if(not contains(outer_map, input))
            {
                outer_map[input] =
                    gen_mod->add_parameter(param_name(param_idx++), input->get_shape());
            }
        }

        // Map pointwise module params to gen module params
        std::unordered_map<instruction_ref, instruction_ref> inner_map;
        auto pm_params = pm->get_parameter_names();
        std::sort(pm_params.begin(), pm_params.end());
        for(std::size_t i = 0; i < ins->inputs().size() and i < pm_params.size(); ++i)
        {
            auto pm_param   = pm->get_parameter(pm_params[i]);
            auto outer_input = ins->inputs()[i];
            inner_map[pm_param] = outer_map.at(outer_input);
        }

        // Copy operations from pointwise module into gen module
        for(auto& pm_ins : *pm)
        {
            if(pm_ins.name() == "@param")
                continue;
            if(pm_ins.name() == "@return")
            {
                std::vector<instruction_ref> ret_inputs;
                for(auto input : pm_ins.inputs())
                    ret_inputs.push_back(inner_map.at(input));
                gen_mod->add_return(ret_inputs);
                continue;
            }
            std::vector<instruction_ref> new_inputs;
            for(auto input : pm_ins.inputs())
                new_inputs.push_back(inner_map.at(input));
            instruction_ref pm_ins_ref = std::find_if(
                pm->begin(), pm->end(), [&](const auto& i) { return &i == &pm_ins; });
            inner_map[pm_ins_ref] =
                gen_mod->add_instruction(pm_ins.get_operator(), new_inputs);
        }

        auto inputs = find_inputs(outer_map, &mpm.get_module(), gen_mod);

        mpm.get_module().replace_instruction(
            ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

struct find_gen_reduce
{
    auto matcher() const
    {
        return match::name("reduce_sum",
                           "reduce_mean",
                           "reduce_max",
                           "reduce_min",
                           "reduce_prod")(match::used_once());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        if(is_already_fused(ins))
            return;

        std::string module_name = "gen_reduce_" + ins->name();
        module_ref gen_mod      = mpm.create_module(module_name);
        gen_mod->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        std::size_t param_idx = 0;
        for(auto input : ins->inputs())
        {
            if(not contains(map_ins, input))
            {
                map_ins[input] =
                    gen_mod->add_parameter(param_name(param_idx++), input->get_shape());
            }
        }

        std::vector<instruction_ref> mod_inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(mod_inputs),
                       [&](auto input) { return map_ins.at(input); });
        gen_mod->add_instruction(ins->get_operator(), mod_inputs);
        auto mod_result = std::prev(gen_mod->end());
        gen_mod->add_return({mod_result});

        auto inputs = find_inputs(map_ins, &mpm.get_module(), gen_mod);

        mpm.get_module().replace_instruction(
            ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

struct find_gen_gather
{
    auto matcher() const { return match::name("gather")(match::used_once()); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        if(is_already_fused(ins))
            return;

        std::string module_name = "gen_gather";
        module_ref gen_mod      = mpm.create_module(module_name);
        gen_mod->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        std::size_t param_idx = 0;
        for(auto input : ins->inputs())
        {
            if(not contains(map_ins, input))
            {
                map_ins[input] =
                    gen_mod->add_parameter(param_name(param_idx++), input->get_shape());
            }
        }

        std::vector<instruction_ref> mod_inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(mod_inputs),
                       [&](auto input) { return map_ins.at(input); });
        gen_mod->add_instruction(ins->get_operator(), mod_inputs);
        auto mod_result = std::prev(gen_mod->end());
        gen_mod->add_return({mod_result});

        auto inputs = find_inputs(map_ins, &mpm.get_module(), gen_mod);

        mpm.get_module().replace_instruction(
            ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

} // namespace

void fuse_gen::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_gen_pointwise{});
    // TODO: Enable when lowering for these ops is implemented
    // match::find_matches(mpm, find_gen_reduce{});
    // match::find_matches(mpm, find_gen_gather{});
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
