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
#include <migraphx/pass_manager.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/op/allocate.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <algorithm>
#include <map>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

std::vector<instruction_ref> get_alloc_aliases(const module& mod)
{
    auto returns = mod.get_returns();
    // Collect all allocation aliases from each return value
    std::vector<instruction_ref> alloc_aliases;
    // Use a join but perhaps a tuple output parameter might be better?
    std::transform(returns.begin(),
                   returns.end(),
                   join_back_inserter(alloc_aliases),
                   [](const auto& i) { return instruction::get_output_alias(i); });
    return alloc_aliases;
}

// Create output parameter names
std::unordered_map<instruction_ref, std::string> create_output_names(const module& mod)
{
    std::unordered_map<instruction_ref, std::string> mod_output_names;
    auto alloc_aliases = get_alloc_aliases(mod);

    std::size_t index = 0;
    if(mod.name().empty())
    {
        // Single return with empty module name: all aliases get "output" or "output_N"
        if(alloc_aliases.size() == 1)
        {
            mod_output_names[alloc_aliases.front()] = "output";
        }
        else
        {
            for(auto ins : alloc_aliases)
            {
                mod_output_names[ins] = "output_" + std::to_string(index++);
            }
        }
    }
    // Preserve main module output buffer naming across migraphx versions
    else
    {
        for(auto ins : alloc_aliases)
        {
            mod_output_names[ins] = param_name(index++, mod.name() + ":#output_");
        }
    }

    return mod_output_names;
}

// Get debug symbols for output parameters from the `return` instruction
std::unordered_map<instruction_ref, std::set<std::string>>
get_output_debug_symbols(const module& mod)
{
    std::unordered_map<instruction_ref, std::set<std::string>> mod_output_debug_symbols;
    auto last_ins = std::prev(mod.end());
    if(mod.has_debug_symbols() and last_ins->name() == "@return" and
       not last_ins->get_debug_symbols().empty())
    {
        auto alloc_aliases = get_alloc_aliases(mod);

        std::size_t index          = 0;
        const auto& output_symbols = last_ins->get_debug_symbols();
        if(alloc_aliases.size() != output_symbols.size())
        {
            migraphx::log::warn()
                << "Size mismatch between output debug symbols and return allocation aliases.";
            return mod_output_debug_symbols;
        }
        for(const auto& os : range(output_symbols.begin(), output_symbols.end()))
        {
            mod_output_debug_symbols[alloc_aliases.at(index)] = {os};
            ++index;
        }
        return mod_output_debug_symbols;
    }
    return mod_output_debug_symbols;
}

// Collect the shape transformations that are applied to the allocation at the end of `path` to
// produce the instruction at the start of it. Instructions that alias their input without changing
// the shape are the identity transformation, so they are skipped. This includes the operator
// writing into the allocation.
std::vector<operation> get_alias_transforms(const std::vector<instruction_ref>& path)
{
    std::vector<operation> ops;
    adjacent_for_each(path.begin(), path.end(), [&](instruction_ref ins, instruction_ref input) {
        if(ins->get_shape() == input->get_shape())
            return;
        auto op = ins->normalized_operator();
        // The descriptor records reshapes with the non-lazy operator
        if(op.name() == "reshape_lazy")
            op = make_op("reshape", {{"dims", ins->get_shape().lens()}});
        ops.push_back(op);
    });
    std::reverse(ops.begin(), ops.end());
    return ops;
}

// Compute the shape that results from applying `ops` to `s`. Every operator must be an alias of
// its input, since a copy is what is being avoided, and it must preserve the number of elements so
// that every element of the buffer is written to exactly once.
optional<shape> compute_alias_view(const std::vector<operation>& ops, const shape& s)
{
    shape result = s;
    for(const auto& op : ops)
    {
        std::vector<shape> inputs = {result};
        if(op.output_alias(inputs).empty())
            return nullopt;
        try
        {
            result = op.compute_shape(inputs);
        }
        catch(const migraphx::exception&)
        {
            return nullopt;
        }
        if(result.elements() != s.elements())
            return nullopt;
    }
    return result;
}

// Compute the transformations that produce `alloc_shape` from `out_shape`, which is the inverse of
// `ops`.
optional<std::vector<operation>> invert_alias_transforms(const shape& alloc_shape,
                                                         const shape& out_shape,
                                                         const std::vector<operation>& ops)
{
    auto inverse = shape_transform_descriptor::create(alloc_shape.lens(), ops).invert();
    if(inverse.empty())
        return nullopt;
    auto result = inverse.generate();
    // Reshapes need to be lazy so the output buffer is aliased instead of copied
    std::transform(result.begin(), result.end(), result.begin(), [](const operation& op) {
        if(op.name() != "reshape")
            return op;
        return make_op("reshape_lazy", op.to_value());
    });
    // The transformations are only the inverse when they produce the same shape the operator
    // writes to, otherwise the buffer would be written to in a different order
    if(compute_alias_view(result, out_shape) != alloc_shape)
        return nullopt;
    return result;
}

// Rather than copying the result into the output buffer, transform the output buffer into the
// shape of the allocation and then write into it directly.
bool replace_alias_allocation(module& m, instruction_ref ins)
{
    auto path = get_alias_path(ins);
    std::vector<instruction_ref> aliases(path.begin(), path.end());
    auto alloc = aliases.back();
    if(alloc->name() != "allocate" or alloc->get_shape().any_of_dynamic())
        return false;
    // Each return value needs its own output parameter, so an allocation that is shared with
    // another return value still needs to be copied
    auto returns = m.get_returns();
    if(std::count_if(returns.begin(), returns.end(), [&](instruction_ref r) {
           return contains(instruction::get_output_alias(r), alloc);
       }) > 1)
        return false;
    auto inverse = invert_alias_transforms(
        alloc->get_shape(), ins->get_shape(), get_alias_transforms(aliases));
    if(not inverse.has_value())
        return false;
    auto out = m.insert_instruction(
        alloc, make_op("allocate", migraphx::value{{"shape", to_value(ins->get_shape())}}));
    out = std::accumulate(
        inverse->begin(), inverse->end(), out, [&](instruction_ref input, const operation& op) {
            return m.insert_instruction(alloc, op, input);
        });
    m.replace_instruction(alloc, out);
    return true;
}

void insert_copy(module& m, const allocation_model& model)
{
    // Rewriting a return can change the aliases of the other returns, so visit them in order
    std::unordered_set<instruction_ref> visited;
    for(auto ins : m.get_returns())
    {
        if(not visited.insert(ins).second)
            continue;
        if(ins->get_shape().any_of_dynamic())
            continue;
        auto aliases = instruction::get_output_alias(ins);
        if(std::any_of(aliases.begin(), aliases.end(), [&](instruction_ref alias) {
               return alias->get_shape() == ins->get_shape();
           }))
            continue;
        if(replace_alias_allocation(m, ins))
            continue;
        auto insert_ins = std::next(ins);
        auto alloc      = m.insert_instruction(
            insert_ins,
            make_op("allocate", migraphx::value{{"shape", to_value(ins->get_shape())}}));
        auto copy = m.insert_instruction(insert_ins, make_op(model.copy()), ins, alloc);
        m.replace_instruction(ins, copy);
    }
}

void insert_submod_allocations(instruction_ref ins, module& mod, const allocation_model& model)
{
    std::vector<instruction_ref> inputs = ins->inputs();
    std::vector<module_ref> mod_args    = ins->module_inputs();

    std::map<std::string, shape> name_shapes;
    for(const auto& smod : mod_args)
    {
        auto ps = smod->get_parameter_shapes();
        name_shapes.insert(ps.begin(), ps.end());
    }

    for(const auto& pn : name_shapes)
    {
        const auto& s = pn.second;
        instruction_ref output{};
        output = mod.insert_instruction(ins, model.allocate(s));
        inputs.push_back(output);
    }

    mod.replace_instruction(ins, ins->get_operator(), inputs, mod_args);
}
} // namespace

void replace_allocate::apply(module_pass_manager& mpm) const
{
    module& m              = mpm.get_module();
    bool is_root           = *mpm.get_root_module() == m;
    bool root_offload_copy = is_root ? this->offload_copy : false;
    // Adjust allocations before replacing
    for(auto ins : iterator_for(m))
    {
        // check if allocations from submodules need to be inserted
        // for now, only the "if" operator is affected
        if(ins->name() != "if")
            continue;
        insert_submod_allocations(ins, m, model);
    }
    if(not root_offload_copy and model.needs_out_params())
        insert_copy(m, model);
    auto mod_output_names         = create_output_names(m);
    auto mod_output_debug_symbols = get_output_debug_symbols(m);
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "allocate")
            continue;

        auto s = ins->get_shape();
        if(not root_offload_copy and model.needs_out_params() and contains(mod_output_names, ins))
        {
            auto out_param = m.add_parameter(mod_output_names[ins], s);
            if(contains(mod_output_debug_symbols, ins))
            {
                m.add_debug_symbols(out_param, mod_output_debug_symbols[ins]);
            }
            m.replace_instruction(ins, out_param);
        }
        else
        {
            m.replace_instruction(ins,
                                  make_op(model.name(), migraphx::value{{"shape", to_value(s)}}));
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
