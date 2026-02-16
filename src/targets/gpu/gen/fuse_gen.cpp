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

// Check if an op is an index transformation (can be fused as input).
// Only ops that the lanewise pass can lower are included.
bool is_index_transform_op(const std::string& name)
{
    return contains({"pad", "gather", "reverse"}, name);
}

// Add an instruction's inputs as gen module params, tracing through
// index transform ops to include them in the gen module.
static void add_gen_inputs(module_ref gen_mod,
                           instruction_ref input,
                           std::unordered_map<instruction_ref, instruction_ref>& outer_map,
                           std::size_t& param_idx)
{
    if(contains(outer_map, input))
        return;

    // If this input is a 1D index transform op with a single user, inline it
    if(is_index_transform_op(input->name()) and input->outputs().size() == 1 and
       input->get_shape().ndim() <= 1)
    {
        // Add the index transform op's inputs as params
        for(auto sub_input : input->inputs())
        {
            add_gen_inputs(gen_mod, sub_input, outer_map, param_idx);
        }
        // Add the index transform op itself into the gen module
        std::vector<instruction_ref> new_inputs;
        for(auto sub_input : input->inputs())
            new_inputs.push_back(outer_map.at(sub_input));
        outer_map[input] = gen_mod->add_instruction(input->get_operator(), new_inputs);
    }
    else
    {
        // Regular input: add as parameter
        outer_map[input] = gen_mod->add_parameter(param_name(param_idx++), input->get_shape());
    }
}

struct find_gen_pointwise
{
    auto matcher() const { return match::name("pointwise")(match::used_once()); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        if(is_already_fused(ins))
            return;

        // Skip multi-output pointwise for now
        if(ins->get_shape().type() == shape::tuple_type)
            return;

        auto* pm = ins->module_inputs().front();

        std::string module_name = "gen_" + pm->name();
        module_ref gen_mod      = mpm.create_module(module_name);
        gen_mod->set_bypass();

        // Map outer instruction inputs to gen module params,
        // tracing through index transform ops to inline them.
        std::unordered_map<instruction_ref, instruction_ref> outer_map;
        std::size_t param_idx = 0;
        for(auto input : ins->inputs())
        {
            add_gen_inputs(gen_mod, input, outer_map, param_idx);
        }

        // Map pointwise module params to gen module values
        std::unordered_map<instruction_ref, instruction_ref> inner_map;
        auto pm_params = pm->get_parameter_names();
        std::sort(pm_params.begin(), pm_params.end());
        for(std::size_t i = 0; i < ins->inputs().size() and i < pm_params.size(); ++i)
        {
            auto pm_param       = pm->get_parameter(pm_params[i]);
            auto outer_input    = ins->inputs()[i];
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
            instruction_ref pm_ins_ref =
                std::find_if(pm->begin(), pm->end(), [&](const auto& i) { return &i == &pm_ins; });
            inner_map[pm_ins_ref] = gen_mod->add_instruction(pm_ins.get_operator(), new_inputs);
        }

        auto inputs = find_inputs(outer_map, &mpm.get_module(), gen_mod);

        mpm.get_module().replace_instruction(ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

struct find_gen_reduce
{
    auto matcher() const
    {
        return match::name("reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod")(
            match::used_once());
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        if(is_already_fused(ins))
            return;

        // Only handle 1D reductions for now
        if(ins->inputs().front()->get_shape().ndim() > 1)
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

        mpm.get_module().replace_instruction(ins, make_op("gpu::gen::op"), inputs, {gen_mod});
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

        mpm.get_module().replace_instruction(ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

// Helper to inline a submodule's operations into a gen module.
// For nested pointwise ops, inlines the pointwise submodule's ops directly.
static void inline_submodule(module_ref gen_mod,
                             const_module_ref pm,
                             const std::vector<instruction_ref>& outer_inputs,
                             std::unordered_map<instruction_ref, instruction_ref>& outer_map)
{
    // Map submodule params to gen module params by sorted name order
    std::unordered_map<instruction_ref, instruction_ref> inner_map;
    auto pm_params = pm->get_parameter_names();
    std::sort(pm_params.begin(), pm_params.end());
    for(std::size_t i = 0; i < outer_inputs.size() and i < pm_params.size(); ++i)
    {
        auto pm_param       = pm->get_parameter(pm_params[i]);
        auto outer_input    = outer_inputs[i];
        inner_map[pm_param] = outer_map.at(outer_input);
    }

    // Copy operations from submodule into gen module
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

        instruction_ref pm_ins_ref =
            std::find_if(pm->begin(), pm->end(), [&](const auto& i) { return &i == &pm_ins; });

        // If this is a pointwise op with a submodule, inline its contents
        if(pm_ins.name() == "pointwise" and not pm_ins.module_inputs().empty())
        {
            auto* pw_mod   = pm_ins.module_inputs().front();
            auto pw_params = pw_mod->get_parameter_names();
            std::sort(pw_params.begin(), pw_params.end());

            // Map pointwise module params to current gen module values
            std::unordered_map<instruction_ref, instruction_ref> pw_map;
            for(std::size_t i = 0; i < pm_ins.inputs().size() and i < pw_params.size(); ++i)
            {
                auto pw_param    = pw_mod->get_parameter(pw_params[i]);
                pw_map[pw_param] = inner_map.at(pm_ins.inputs()[i]);
            }

            // Inline the pointwise ops
            instruction_ref pw_result;
            for(auto& pw_ins : *pw_mod)
            {
                if(pw_ins.name() == "@param")
                    continue;
                if(pw_ins.name() == "@return")
                {
                    pw_result = pw_map.at(pw_ins.inputs().front());
                    continue;
                }
                std::vector<instruction_ref> pw_inputs;
                for(auto input : pw_ins.inputs())
                    pw_inputs.push_back(pw_map.at(input));
                instruction_ref pw_ins_ref = std::find_if(
                    pw_mod->begin(), pw_mod->end(), [&](const auto& i) { return &i == &pw_ins; });
                pw_map[pw_ins_ref] = gen_mod->add_instruction(pw_ins.get_operator(), pw_inputs);
            }
            inner_map[pm_ins_ref] = pw_result;
            continue;
        }

        std::vector<instruction_ref> new_inputs;
        for(auto input : pm_ins.inputs())
            new_inputs.push_back(inner_map.at(input));
        inner_map[pm_ins_ref] = gen_mod->add_instruction(pm_ins.get_operator(), new_inputs);
    }
}

struct find_gen_fused_reduce
{
    auto matcher() const { return match::name("fused_reduce")(match::used_once()); }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        if(is_already_fused(ins))
            return;

        // Only handle 1D reductions for now (input.ndim <= 1)
        if(ins->inputs().front()->get_shape().ndim() > 1)
            return;

        auto* pm = ins->module_inputs().front();

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

        inline_submodule(gen_mod, pm, ins->inputs(), outer_map);

        auto inputs = find_inputs(outer_map, &mpm.get_module(), gen_mod);
        mpm.get_module().replace_instruction(ins, make_op("gpu::gen::op"), inputs, {gen_mod});
    }
};

} // namespace

void fuse_gen::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_gen_pointwise{});
    match::find_matches(mpm, find_gen_reduce{});
    // TODO: Enable fused_reduce when the lowering handles
    // pointwise ops inside the reduction loop
    // match::find_matches(mpm, find_gen_fused_reduce{});
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
