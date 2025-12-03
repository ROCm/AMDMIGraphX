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
#include <migraphx/gpu/gen/fuse_gen.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/param_utils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace gen {

// Check if shape has any broadcasted dimensions
static bool has_broadcast(const shape& s)
{
    const auto& strides = s.strides();
    return std::any_of(strides.begin(), strides.end(), [](auto st) { return st == 0; });
}

// Operations that can be fused into the gen module as input transformations
// Note: For now, we disable input fusion because the codegen (generate_pointwise_kernel)
// doesn't properly handle operations like transpose, slice, pad, etc. inside the module.
// These will be enabled once proper lowering support is added.
static const std::unordered_set<std::string>& fusable_input_ops()
{
    static const std::unordered_set<std::string> names = {};
    return names;
}

// Check if an operation is a fusable input operation
static bool is_fusable_input_op(const std::string& name)
{
    return contains(fusable_input_ops(), name);
}

// Check if instruction is used only once
static bool is_used_once(instruction_ref ins) { return ins->outputs().size() == 1; }

// Get the chain of fusable input operations from an input
// Only includes operations that are used once
static std::tuple<instruction_ref, std::vector<operation>>
get_fusable_input_op_stream(instruction_ref lower_input)
{
    instruction_ref upper_input = lower_input;
    std::vector<operation> op_stream;
    while(is_fusable_input_op(upper_input->name()) and is_used_once(upper_input))
    {
        operation op = upper_input->get_operator();
        op_stream.push_back(op);
        upper_input = upper_input->inputs().at(0);
    }
    return {upper_input, op_stream};
}

// Fuse input operations into a module
static void fuse_gen_input_ops(module_ref mm,
                               const std::vector<instruction_ref>& inputs,
                               std::unordered_map<instruction_ref, instruction_ref>* map_ins)
{
    assert(map_ins != nullptr);
    size_t input_cnt = mm->get_parameters().size();
    for(instruction_ref input : inputs)
    {
        if(contains(*map_ins, input))
            continue;
        auto [upper_input, op_stream] = get_fusable_input_op_stream(input);
        if(not contains(*map_ins, upper_input))
            (*map_ins)[upper_input] =
                mm->add_parameter(param_name(input_cnt++), upper_input->get_shape().as_standard());
        instruction_ref prev_input = (*map_ins)[upper_input];
        for(const auto& op : reverse(op_stream))
        {
            prev_input = mm->add_instruction(op, {prev_input});
        }
        (*map_ins)[input] = prev_input;
    }
}

// Check if an operation can be handled by gen IR
// Note: fused_reduce is not yet supported - the lowering to gen IR operators
// (strided_load, dpp_reduce, etc.) is not complete yet
static bool is_gen_supported(instruction_ref ins)
{
    static const std::unordered_set<std::string> supported_ops = {"pointwise"};
    return supported_ops.count(ins->name()) > 0;
}

// Check if any input has fusable input ops in its chain
static bool has_fusable_inputs(instruction_ref ins)
{
    for(auto input : ins->inputs())
    {
        if(is_fusable_input_op(input->name()))
            return true;
    }
    return false;
}

// Try to fuse an operation into gen IR
static bool try_fuse_op(module& m, module_pass_manager& mpm, instruction_ref ins)
{
    if(not is_gen_supported(ins))
        return false;

    // Skip if no module inputs
    if(ins->module_inputs().empty())
        return false;

    // Skip multi-output operations (tuple outputs) - not supported yet
    if(ins->get_shape().type() == shape::tuple_type)
        return false;

    // For pointwise: skip if any input or output has broadcasted dimensions (stride 0)
    if(ins->name() == "pointwise")
    {
        if(has_broadcast(ins->get_shape()))
            return false;

        auto out_type = ins->get_shape().type();
        for(auto input : ins->inputs())
        {
            if(has_broadcast(input->get_shape()))
                return false;
            // Skip if input and output types differ (e.g., bit_cast)
            if(input->get_shape().type() != out_type)
                return false;
        }
    }

    // Get the original submodule
    auto* orig_mod = ins->module_inputs().front();

    // Check if we need to fuse input operations
    bool needs_input_fusion = has_fusable_inputs(ins);

    std::vector<instruction_ref> new_inputs;
    module_ref gen_mod = nullptr;

    if(needs_input_fusion)
    {
        // Create a new module that includes fused input operations
        std::string mod_name =
            "gen_" + ins->name() + "_" + std::to_string(ins->get_shape().elements());
        if(m.name() != "main")
            mod_name = m.name() + ":" + mod_name;

        gen_mod = mpm.create_module(mod_name);
        gen_mod->set_bypass();

        // Fuse input operations into the new module
        std::unordered_map<instruction_ref, instruction_ref> ins_map;
        fuse_gen_input_ops(gen_mod, ins->inputs(), &ins_map);

        // Fuse the original submodule into gen_mod
        auto fused_outputs = gen_mod->fuse(*orig_mod, ins->inputs(), &ins_map);
        gen_mod->add_return(fused_outputs);

        // Collect the new inputs (the top-level inputs after walking up fusable chains)
        std::unordered_set<instruction_ref> seen;
        for(auto input : ins->inputs())
        {
            auto [upper_input, op_stream] = get_fusable_input_op_stream(input);
            if(not contains(seen, upper_input))
            {
                new_inputs.push_back(upper_input);
                seen.insert(upper_input);
            }
        }
    }
    else
    {
        // No input fusion needed, use original inputs and module
        new_inputs = ins->inputs();
        gen_mod    = const_cast<module*>(orig_mod);
    }

    // Create allocation for output
    auto alloc =
        m.insert_instruction(ins, make_op("allocate", {{"shape", to_value(ins->get_shape())}}));
    new_inputs.push_back(alloc);

    // Create gpu::gen::op wrapped in precompile_op
    if(needs_input_fusion)
    {
        m.replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(make_op("gpu::gen::op"))}}),
            new_inputs,
            {gen_mod});
    }
    else
    {
        m.replace_instruction(
            ins,
            make_op("gpu::precompile_op", {{"op", to_value(make_op("gpu::gen::op"))}}),
            new_inputs,
            ins->module_inputs());
    }
    return true;
}

void fuse_gen::apply(module_pass_manager& mpm) const
{
    auto& m = mpm.get_module();

    for(auto ins : iterator_for(m))
    {
        try_fuse_op(m, mpm, ins);
    }
}

} // namespace gen
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
