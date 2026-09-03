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

#include <migraphx/split_seq_len.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

const std::string padded_suffix = "#padded";

struct seq_split_info
{
    // The sequence-length variable shared by every non-fixed parameter dimension
    sym::expr seq;
    std::size_t min = 0;
    std::size_t max = 0;
    // Parameters that carry the symbolic sequence dimension, sorted by name
    std::vector<std::string> seq_params;
};

bool contains_op(const_module_ref mm, const std::string& name)
{
    return std::any_of(mm->begin(), mm->end(), [&](const auto& ins) { return ins.name() == name; });
}

/**
 * A kv-cache model whose only non-fixed dimension is the current sequence length: every
 * non-fixed symbolic parameter dimension is the same variable, and that variable is the
 * sequence dimension of a concat_past_present kv-cache update.
 */
optional<seq_split_info> find_seq_split(const_module_ref mm)
{
    // Applying twice is a no-op
    if(not contains_op(mm, "concat_past_present") or contains_op(mm, "select_module"))
        return nullopt;
    seq_split_info info;
    auto param_names = mm->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    for(const auto& name : param_names)
    {
        auto s = mm->get_parameter_shape(name);
        if(not s.dynamic())
            continue;
        if(not s.symbolic())
            return nullopt;
        bool carries_seq = false;
        for(const auto& dd : s.dyn_dims())
        {
            if(dd.is_fixed())
                continue;
            if(dd.sym_expr.name() != "variable")
                return nullopt;
            if(info.seq.empty())
                info.seq = dd.sym_expr;
            else if(info.seq != dd.sym_expr)
                return nullopt;
            carries_seq = true;
        }
        if(carries_seq)
            info.seq_params.push_back(name);
    }
    if(info.seq.empty())
        return nullopt;
    // The variable must be the sequence dimension of the kv-cache update: dimension 2 of
    // the {batch, kv_heads, seq, head_size} current k/v input of a concat_past_present.
    bool is_seq_len =
        std::any_of(iterator_for(*mm).begin(), iterator_for(*mm).end(), [&](auto ins) {
            if(ins->name() != "concat_past_present")
                return false;
            const auto& s = ins->inputs().front()->get_shape();
            if(not s.symbolic() or s.ndim() != 4)
                return false;
            const auto& dd = s.dyn_dims()[2];
            return not dd.is_fixed() and dd.sym_expr == info.seq;
        });
    if(not is_seq_len)
        return nullopt;
    auto interval = shape::dynamic_dimension{info.seq}.get_interval();
    info.min      = interval.min;
    info.max      = interval.max;
    if(info.min >= info.max)
        return nullopt;
    return info;
}

/**
 * Inserter that resolves the symbolic-attribute ops while cloning into a submodule where
 * the sequence length is the fixed value bound in symbol_map. Everything else clones
 * unchanged and becomes static through ordinary shape propagation from the static
 * parameters.
 */
module::inserter make_static_inserter(const std::unordered_map<sym::expr, std::size_t>& symbol_map)
{
    auto resolve_dim = [=](const shape::dynamic_dimension& dd) -> std::size_t {
        if(dd.is_fixed())
            return dd.get_interval().min;
        return dd.sym_expr.eval_uint(symbol_map);
    };
    return [=](module& m,
               instruction_ref,
               const operation& op,
               const std::vector<instruction_ref>& inputs,
               const std::vector<module_ref>& mod_args) -> instruction_ref {
        if(op.name() == "dimensions_of" and not inputs.front()->get_shape().dynamic())
        {
            auto v           = op.to_value();
            auto start       = v.at("start").to<std::size_t>();
            auto end         = v.at("end").to<std::size_t>();
            const auto& lens = inputs.front()->get_shape().lens();
            std::vector<int64_t> dims(lens.begin() + start, lens.begin() + end);
            return m.add_literal(literal{shape{shape::int64_type, {end - start}}, dims});
        }
        if(op.name() == "dynamic_range" and
           std::all_of(inputs.begin(), inputs.end(), [](auto ins) { return ins->can_eval(); }))
        {
            instruction_ref lit;
            visit_all(inputs[0]->eval(),
                      inputs[1]->eval(),
                      inputs[2]->eval())([&](auto start, auto limit, auto delta) {
                auto start_val = start.front();
                auto delta_val = delta.front();
                double n =
                    std::ceil((double(limit.front()) - double(start_val)) / double(delta_val));
                std::size_t nelements = n > 0 ? std::size_t(n) : 0;
                std::vector<decltype(start_val)> vals(nelements);
                std::generate(vals.begin(), vals.end(), [&] {
                    auto result = start_val;
                    start_val += delta_val;
                    return result;
                });
                lit =
                    m.add_literal(literal{shape{inputs[0]->get_shape().type(), {nelements}}, vals});
            });
            return lit;
        }
        if(op.name() == "multibroadcast")
        {
            auto v = op.to_value();
            if(not v.at("out_dyn_dims").empty())
            {
                std::vector<shape::dynamic_dimension> dds;
                migraphx::from_value(v.at("out_dyn_dims"), dds);
                std::vector<std::size_t> out_lens(dds.size());
                std::transform(dds.begin(), dds.end(), out_lens.begin(), resolve_dim);
                // The extra shape-donor inputs only carry dynamic dimensions; the static
                // target does not need them.
                return m.add_instruction(make_op("multibroadcast", {{"out_lens", out_lens}}),
                                         inputs.front());
            }
        }
        if(op.name() == "min" and inputs.size() == 2)
        {
            // Fold min(max(x, a), b) to b when b <= a. With the sequence length fixed at
            // the cache size this proves the clamped past length of a padded prefill is
            // zero, which lets the positions and causal mask constant-fold.
            auto is_scalar_integral = [](instruction_ref ins) {
                return ins->get_shape().elements() == 1 and
                       shape::is_integral(ins->get_shape().type());
            };
            auto fold_clamp = [&](instruction_ref max_ins,
                                  instruction_ref bound) -> optional<int64_t> {
                if(max_ins->name() != "max" or not bound->can_eval() or
                   not is_scalar_integral(bound))
                    return nullopt;
                auto lower_it = std::find_if(
                    max_ins->inputs().begin(), max_ins->inputs().end(), [&](auto input) {
                        return input->can_eval() and is_scalar_integral(input);
                    });
                if(lower_it == max_ins->inputs().end())
                    return nullopt;
                int64_t lower_value = 0;
                int64_t bound_value = 0;
                (*lower_it)->eval().visit([&](auto data) { lower_value = int64_t(data.front()); });
                bound->eval().visit([&](auto data) { bound_value = int64_t(data.front()); });
                if(bound_value > lower_value)
                    return nullopt;
                return bound_value;
            };
            for(auto perm :
                {std::make_pair(inputs[0], inputs[1]), std::make_pair(inputs[1], inputs[0])})
            {
                auto folded = fold_clamp(perm.first, perm.second);
                if(folded.has_value())
                {
                    return m.add_literal(literal{
                        shape{perm.second->get_shape().type(), perm.second->get_shape().lens()},
                        {*folded}});
                }
            }
        }
        if(op.name() == "eval_expr_from_shape")
        {
            auto v = op.to_value();
            std::vector<sym::expr> expressions;
            migraphx::from_value(v.at("expressions"), expressions);
            std::vector<int64_t> vals(expressions.size());
            std::transform(expressions.begin(),
                           expressions.end(),
                           vals.begin(),
                           [&](const auto& e) { return int64_t(e.eval_uint(symbol_map)); });
            return m.add_literal(literal{shape{shape::int64_type, {vals.size()}}, vals});
        }
        if(op.name() == "allocate")
        {
            auto v = op.to_value();
            if(v.contains("shape") and not v.at("shape").is_null())
            {
                shape alloc_shape;
                migraphx::from_value(v.at("shape"), alloc_shape);
                if(alloc_shape.any_of_dynamic())
                {
                    return m.add_instruction(
                        make_op("allocate",
                                {{"shape", to_value(alloc_shape.to_static(symbol_map))}}),
                        inputs);
                }
            }
        }
        return m.add_instruction(op, inputs, mod_args);
    };
}

} // namespace

void split_seq_len::apply(module_pass_manager& mpm) const
{
    module_ref mm = &mpm.get_module();
    auto info     = find_seq_split(mm);
    if(not info.has_value())
        return;

    auto param_names = mm->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    auto original_output_shapes = mm->get_output_shapes();

    // Every trimmed output dimension must be computable from the sequence parameters at
    // run time, checked up front so the module is not left half rewritten.
    std::unordered_set<sym::expr> known_variables;
    for(const auto& name : info->seq_params)
    {
        for(const auto& dd : mm->get_parameter_shape(name).dyn_dims())
        {
            if(dd.sym_expr.name() == "variable")
                known_variables.insert(sym::as_symbol(dd.sym_expr));
        }
    }
    for(const auto& os : original_output_shapes)
    {
        if(not os.dynamic())
            continue;
        if(not os.symbolic())
            return;
        for(const auto& dd : os.dyn_dims())
        {
            if(dd.is_fixed())
                continue;
            auto variables = sym::find_variables(dd.sym_expr);
            if(not std::all_of(variables.begin(), variables.end(), [&](const auto& v) {
                   return contains(known_variables, v);
               }))
                return;
        }
    }

    // Build one static submodule per sequence length. The maximum-length submodule reads
    // the zero-padded inputs, so it serves every length the decode submodule does not.
    auto build_submodule = [&](std::size_t seq_len, bool padded) {
        auto* submod = mpm.create_module("seq_len_" + std::to_string(seq_len));
        std::unordered_map<sym::expr, std::size_t> symbol_map{{info->seq, seq_len}};
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        for(const auto& name : param_names)
        {
            auto param        = mm->get_parameter(name);
            auto static_shape = param->get_shape().to_static(symbol_map);
            auto param_name =
                (padded and contains(info->seq_params, name)) ? name + padded_suffix : name;
            map_ins[param] = submod->add_parameter(param_name, static_shape);
        }
        auto outputs = submod->add_instructions(mm, &map_ins, make_static_inserter(symbol_map));
        submod->add_return(outputs);
        return submod;
    };
    module_ref decode_mod  = build_submodule(info->min, false);
    module_ref prefill_mod = build_submodule(info->max, true);

    // The submodules must be fully static to be worth dispatching to
    auto all_static = [](const_module_ref submod) {
        return std::all_of(submod->begin(), submod->end(), [](const auto& ins) {
            return not ins.get_shape().any_of_dynamic();
        });
    };
    if(not all_static(decode_mod) or not all_static(prefill_mod))
        return;

    // Pass every parameter plus a padded copy of each sequence-length parameter; the
    // param_names attribute lets each submodule bind only the arguments it declares.
    std::vector<instruction_ref> sm_inputs;
    std::vector<std::string> sm_param_names;
    for(const auto& name : param_names)
    {
        sm_inputs.push_back(mm->get_parameter(name));
        sm_param_names.push_back(name);
    }
    std::vector<instruction_ref> seq_param_ins;
    for(const auto& name : info->seq_params)
    {
        auto param = mm->get_parameter(name);
        seq_param_ins.push_back(param);
        sm_inputs.push_back(mm->add_instruction(make_op("fixed_pad"), param));
        sm_param_names.push_back(name + padded_suffix);
    }

    shape out_attr{original_output_shapes};
    auto sm_ins = mm->add_instruction(make_op("select_module",
                                              {{"output_dyn_shapes", to_value(out_attr)},
                                               {"param_names", to_value(sm_param_names)}}),
                                      sm_inputs,
                                      {decode_mod, prefill_mod});

    std::vector<instruction_ref> outputs;
    for(auto i : range(original_output_shapes.size()))
    {
        auto elem      = mm->add_instruction(make_op("get_tuple_elem", {{"index", i}}), sm_ins);
        const auto& os = original_output_shapes[i];
        // Trim the axes the prefill submodule padded back to the actual sequence length
        std::vector<int64_t> axes;
        std::vector<sym::expr> ends;
        if(os.dynamic())
        {
            for(auto axis : range(os.ndim()))
            {
                const auto& dd = os.dyn_dims()[axis];
                if(dd.is_fixed())
                    continue;
                axes.push_back(int64_t(axis));
                ends.push_back(dd.sym_expr);
            }
        }
        if(axes.empty())
        {
            outputs.push_back(elem);
            continue;
        }
        auto ends_input = mm->add_instruction(
            make_op("eval_expr_from_shape", {{"expressions", to_value(ends)}}), seq_param_ins);
        std::vector<sym::expr> starts(axes.size(), sym::lit(int64_t{0}));
        auto starts_input = mm->add_literal(
            literal{shape{shape::int64_type, {axes.size()}}, std::vector<int64_t>(axes.size(), 0)});
        outputs.push_back(mm->add_instruction(
            make_op("dyn_slice",
                    {{"axes", axes}, {"starts", to_value(starts)}, {"ends", to_value(ends)}}),
            elem,
            starts_input,
            ends_input));
    }
    mm->replace_return(outputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
