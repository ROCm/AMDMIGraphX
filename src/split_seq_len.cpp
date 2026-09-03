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
#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

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
    // Applying twice is a no-op. eval_expr_from_shape needs a symbolic input to bind its
    // variables, so it cannot be cloned over the static submodule parameters.
    if(not contains_op(mm, "concat_past_present") or contains_op(mm, "select_module") or
       contains_op(mm, "eval_expr_from_shape"))
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

    // select_module binds each submodule's parameters, sorted by name, to the leading
    // arguments in order. The arguments are laid out as the shared parameters, then a
    // zero-padded copy of each sequence-length parameter, then the sequence-length
    // parameters themselves, and every submodule parameter is named by its argument
    // position with param_name so the sorted names follow that layout. The decode
    // submodule declares every argument and so matches only when the sequence-length
    // arguments have exactly its length; the prefill submodule declares just the
    // leading arguments and reads the padded copies, so it matches every other length.
    std::vector<std::string> shared_params;
    std::copy_if(param_names.begin(),
                 param_names.end(),
                 std::back_inserter(shared_params),
                 [&](const auto& name) { return not contains(info->seq_params, name); });
    const auto padded_index = [&](std::size_t j) { return shared_params.size() + j; };
    const auto raw_index    = [&](std::size_t j) {
        return shared_params.size() + info->seq_params.size() + j;
    };
    const std::unordered_map<sym::expr, std::size_t> max_symbol_map{{info->seq, info->max}};

    auto build_submodule = [&](std::size_t seq_len, bool padded) {
        auto* submod = mpm.create_module("seq_len_" + std::to_string(seq_len));
        std::unordered_map<sym::expr, std::size_t> symbol_map{{info->seq, seq_len}};
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        for(auto i : range(shared_params.size()))
        {
            auto param = mm->get_parameter(shared_params[i]);
            map_ins[param] =
                submod->add_parameter(param_name(i), param->get_shape().to_static(symbol_map));
        }
        for(auto j : range(info->seq_params.size()))
        {
            auto param        = mm->get_parameter(info->seq_params[j]);
            auto padded_param = submod->add_parameter(param_name(padded_index(j)),
                                                      param->get_shape().to_static(max_symbol_map));
            if(padded)
                map_ins[param] = padded_param;
        }
        if(not padded)
        {
            for(auto j : range(info->seq_params.size()))
            {
                auto param     = mm->get_parameter(info->seq_params[j]);
                map_ins[param] = submod->add_parameter(param_name(raw_index(j)),
                                                       param->get_shape().to_static(symbol_map));
            }
        }
        // The clone keeps the dynamic ops as they are; simplify_dyn_ops resolves them once
        // their inputs are static.
        auto outputs = submod->add_instructions(mm, &map_ins);
        submod->add_return(outputs);
        return submod;
    };
    module_ref decode_mod  = build_submodule(info->min, false);
    module_ref prefill_mod = build_submodule(info->max, true);

    std::vector<instruction_ref> sm_inputs;
    std::transform(shared_params.begin(),
                   shared_params.end(),
                   std::back_inserter(sm_inputs),
                   [&](const auto& name) { return mm->get_parameter(name); });
    std::vector<instruction_ref> seq_param_ins;
    std::transform(info->seq_params.begin(),
                   info->seq_params.end(),
                   std::back_inserter(seq_param_ins),
                   [&](const auto& name) { return mm->get_parameter(name); });
    std::transform(seq_param_ins.begin(),
                   seq_param_ins.end(),
                   std::back_inserter(sm_inputs),
                   [&](auto param) { return mm->add_instruction(make_op("fixed_pad"), param); });
    sm_inputs.insert(sm_inputs.end(), seq_param_ins.begin(), seq_param_ins.end());

    shape out_attr{original_output_shapes};
    auto sm_ins =
        mm->add_instruction(make_op("select_module", {{"output_dyn_shapes", to_value(out_attr)}}),
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
