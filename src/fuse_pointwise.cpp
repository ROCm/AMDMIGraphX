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
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/rewrite_reshapes.hpp>
#include <migraphx/rewrite_broadcasts.hpp>
#include <iterator>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_POINTWISE_FUSION)

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static literal get_scalar(instruction_ref ins)
{
    if(contains({"contiguous", "broadcast", "multibroadcast"}, ins->name()))
        return get_scalar(ins->inputs().front());
    const auto& s = ins->get_shape();
    if(s.dynamic() or (s.elements() != 1 and not(s.scalar())))
        return {};
    if(not ins->can_eval())
        return {};
    auto e = ins->eval();
    literal r{};
    // needed for bool as visit_at invokes as() which promotes bool to int8
    // Without this we'll break type checks for logical ops that are fused.
    if(e.get_shape().type() == shape::bool_type)
    {
        r = literal{e.at<bool>()};
    }
    else
    {
        e.visit_at([&](auto x) { r = literal{x}; });
    }
    return r;
}

static shape to_scalar(const shape& s) { return shape{s.type()}; }

static bool is_dead(instruction_ref ins)
{
    if(ins->name() == "@return")
        return false;
    if(ins->outputs().empty())
        return true;
    if(ins->name() != "pointwise")
        return false;
    return ends_with(ins->module_inputs().front()->name(), "-deleted");
}

// We dont want to consider the `extra` instruction as dead as it might be an implicit return
static bool is_used_once(instruction_ref ins, const instruction_ref* extra = nullptr)
{
    return std::count_if(ins->outputs().begin(), ins->outputs().end(), [&](auto output) {
               if(extra and *extra == output)
                   return true;
               return not is_dead(output);
           }) == 1;
}

static void create_pointwise_modules(module_pass_manager& mpm)
{
    std::size_t n = 0;
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(not ins->get_operator().attributes().get("pointwise", false))
            continue;
        if(ins->get_operator().name() == "layout")
            continue;
        auto* pm = mpm.create_module(mpm.get_module().name() + ":pointwise" + std::to_string(n++));
        pm->set_bypass();

        std::unordered_map<instruction_ref, instruction_ref> param_map;
        std::vector<instruction_ref> pointwise_inputs;
        std::size_t i = 0;

        for(auto input : ins->inputs())
        {
            if(contains(param_map, input))
                continue;
            auto scalar = get_scalar(input);
            // Have dynamic shapes always get put into a pointwise module even if scalar input
            if(scalar.empty() or input->get_shape().dynamic())
            {
                pointwise_inputs.push_back(input);
                param_map[input] =
                    pm->add_parameter(param_name(i), shape{input->get_shape().type()});
                i++;
            }
            else
            {
                param_map[input] = pm->add_literal(scalar);
            }
        }

        // Don't create pointwise module if no inputs are detected
        if(pointwise_inputs.empty())
            continue;

        std::vector<instruction_ref> inputs;
        std::transform(ins->inputs().begin(),
                       ins->inputs().end(),
                       std::back_inserter(inputs),
                       [&](auto input) { return param_map[input]; });
        auto r = pm->add_instruction(ins->get_operator(), inputs);
        pm->add_return({r});

        mpm.get_module().replace_instruction(ins, make_op("pointwise"), pointwise_inputs, {pm});
    }
}

static module::with_inputs append_pointwise_module(instruction_ref ins, instruction_ref output)
{
    std::unordered_set<instruction_ref> original_inputs{ins->inputs().begin(), ins->inputs().end()};
    original_inputs.insert(output->inputs().begin(), output->inputs().end());
    module pm     = *ins->module_inputs().at(0);
    const_module_ref xm  = output->module_inputs().at(0);
    const bool dependent = contains(output->inputs(), ins);
    assert(not dependent or pm.get_returns().size() == 1);

    std::unordered_map<instruction_ref, instruction_ref> map_ins =
        pm.get_ins_param_map(ins->inputs());
    if(dependent)
        map_ins[ins] = pm.get_returns().front();
    auto returns = pm.fuse(*xm, output->inputs(), &map_ins, nullptr, &to_scalar);
    if(not is_used_once(ins, &output) or not dependent)
    {
        auto ireturns = pm.get_returns();
        returns.insert(returns.end(), ireturns.begin(), ireturns.end());
    }
    pm.replace_return(returns);
    auto inputs = find_inputs(map_ins, original_inputs, &pm);
    return {std::move(pm), inputs};
}

static void replace_with_tuple(module& m, instruction_ref ins, instruction_ref rep, bool first)

{
    if(rep->get_shape().type() != shape::tuple_type)
    {
        assert(ins->get_shape().type() != shape::tuple_type);
        m.replace_instruction(ins, rep);
        return;
    }
    if(ins->get_shape().type() != shape::tuple_type)
    {
        auto i = first ? 0 : rep->get_shape().sub_shapes().size() - 1;
        auto elem =
            m.insert_instruction(std::next(rep), make_op("get_tuple_elem", {{"index", i}}), rep);
        m.replace_instruction(ins, elem);
        return;
    }
    // TODO: We need to add a new operator to repack a tuple to support this scenario
    if(std::any_of(ins->outputs().begin(), ins->outputs().end(), [](instruction_ref output) {
           return output->name() != "get_tuple_elem";
       }))
        MIGRAPHX_THROW("Unsupported tuple replacement");
    std::size_t start =
        first ? 0 : rep->get_shape().sub_shapes().size() - ins->get_shape().sub_shapes().size();
    auto outputs = ins->outputs();
    for(auto output : outputs)
    {
        auto v = output->get_operator().to_value();
        auto i = v.at("index").to<std::size_t>();
        assert((i + start) < rep->get_shape().sub_shapes().size());
        m.replace_instruction(output, make_op("get_tuple_elem", {{"index", i + start}}), rep);
    }
}

static instruction_ref
merge_instruction(module_pass_manager& mpm, instruction_ref input, instruction_ref output)
{
    auto fused = append_pointwise_module(input, output);
    auto name  = fused.mod.name();
    mpm.rename_module(name, name + ":" + output->module_inputs().front()->name() + "-deleted");
    auto* new_pm = mpm.create_module(name, std::move(fused.mod));
    auto fins =
        mpm.get_module().insert_instruction(output, input->get_operator(), fused.inputs, {new_pm});
    if(fins->get_shape().tuple_size() != output->get_shape().tuple_size())
    {
        mpm.get_module().move_output_instructions_after(input, fins);
        replace_with_tuple(mpm.get_module(), input, fins, false);
    }
    replace_with_tuple(mpm.get_module(), output, fins, true);
    return fins;
}

static auto find_input_pointwise(const module& m, instruction_ref ins, bool multi_out)
{
    auto it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
        return i->name() == "pointwise" and i->outputs().size() == 1 and m.has_instruction(i);
    });
    if(it == ins->inputs().end() and multi_out)
    {
        it = std::find_if(ins->inputs().begin(), ins->inputs().end(), [&](auto i) {
            if(not m.has_instruction(i))
                return false;
            auto base_distance = std::distance(i, ins);
            return i->name() == "pointwise" and
                   std::none_of(i->outputs().begin(), i->outputs().end(), [&](auto output) {
                       if(not m.has_instruction(output))
                           return true;
                       if(output == ins)
                           return false;
                       if(std::distance(i, output) > base_distance)
                           return false;
                       return reaches(output, ins, &m);
                   });
        });
    }
    return it;
}

static std::vector<instruction_ref>
find_output_pointwise(const module& m, instruction_ref ins, bool multi_out)
{
    std::vector<instruction_ref> result;
    if(not multi_out)
        return result;
    std::vector<instruction_ref> outputs;
    std::copy_if(ins->outputs().begin(),
                 ins->outputs().end(),
                 std::back_inserter(outputs),
                 [&](instruction_ref output) {
                     if(output->name() != "pointwise")
                         return false;
                     if(not m.has_instruction(output))
                         return false;
                     if(is_dead(output))
                         return false;
                     return true;
                 });
    if(outputs.size() < 2)
        return result;
    std::sort(outputs.begin(), outputs.end(), by(std::less<>{}, [&](auto x) {
                  return std::distance(ins, x);
              }));
    std::copy_if(outputs.begin(), outputs.end(), std::back_inserter(result), [&](auto output) {
        return std::none_of(
            result.begin(), result.end(), [&](auto other) { return reaches(other, output, &m); });
    });
    return result;
}

// A later pass (such as eliminate_common_subexpression) can merge two operands of
// a pointwise instruction into the same instruction. This leaves the pointwise with
// duplicate operands while its submodule still has a distinct parameter for each
// original operand, breaking the invariant that every pointwise has exactly one
// parameter per distinct input. Rebuild any such pointwise so that each distinct
// operand maps to a single parameter.
static bool dedup_pointwise_inputs(module_pass_manager& mpm)
{
    bool changed = false;
    auto& m      = mpm.get_module();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "pointwise")
            continue;
        auto inputs = ins->inputs();
        std::unordered_set<instruction_ref> seen;
        std::vector<instruction_ref> deduped;
        std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(deduped), [&](auto input) {
            return seen.insert(input).second;
        });
        if(deduped.size() == inputs.size())
            continue;

        const_module_ref sm = ins->module_inputs().front();
        module pm;
        pm.set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        auto returns = pm.fuse(*sm, inputs, &map_ins, nullptr, &to_scalar);
        pm.add_return(returns);
        auto* new_pm = mpm.create_module(sm->name() + ":dedup", std::move(pm));
        new_pm->set_bypass();
        m.replace_instruction(ins, make_op("pointwise"), deduped, {new_pm});
        changed = true;
    }
    return changed;
}

static bool split_pointwise_through_slices(module_pass_manager& mpm)
{
    bool changed    = false;
    auto& m         = mpm.get_module();
    std::size_t idx = 0;

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "pointwise")
            continue;
        if(ins->get_shape().type() == shape::tuple_type)
            continue;

        auto outputs = ins->outputs();
        if(outputs.size() < 2)
            continue;

        // All consumers must be slice instructions
        if(not all_of(outputs, [](instruction_ref output) {
               return output->name() == "slice" and output->inputs().size() == 1;
           }))
            continue;

        // Cache slice values to avoid repeated to_value() calls
        std::unordered_map<instruction_ref, value> slice_vals;
        for(auto output : outputs)
            slice_vals[output] = output->get_operator().to_value();

        // All slices must be on the same single axis
        auto axes = slice_vals[outputs.front()]["axes"].to_vector<int64_t>();
        if(axes.size() != 1)
            continue;
        if(not all_of(outputs, [&](instruction_ref output) {
               return slice_vals[output]["axes"].to_vector<int64_t>() == axes;
           }))
            continue;

        auto get_starts = [&](instruction_ref s) {
            return slice_vals[s]["starts"].to_vector<int64_t>()[0];
        };
        auto get_ends = [&](instruction_ref s) {
            return slice_vals[s]["ends"].to_vector<int64_t>()[0];
        };

        // Sort slices by start position and check for no overlap
        std::sort(outputs.begin(), outputs.end(), by(std::less<>{}, get_starts));
        if(std::adjacent_find(
               outputs.begin(), outputs.end(), [&](instruction_ref a, instruction_ref b) {
                   return get_starts(b) < get_ends(a);
               }) != outputs.end())
            continue;

        // At least one slice consumer must feed into a pointwise op
        if(none_of(outputs, [](instruction_ref s) {
               return any_of(s->outputs(),
                             [](instruction_ref c) { return c->name() == "pointwise"; });
           }))
            continue;

        // Split: replace each slice with a pointwise on sliced inputs
        auto* src_pm = ins->module_inputs().front();
        auto pm_name = src_pm->name();
        auto inputs  = ins->inputs();
        for(const auto& slice_ins : outputs)
        {
            auto slice_op = slice_ins->get_operator();

            std::vector<instruction_ref> sliced_inputs;
            sliced_inputs.reserve(inputs.size());
            transform(inputs, std::back_inserter(sliced_inputs), [&](instruction_ref input) {
                return m.insert_instruction(slice_ins, slice_op, input);
            });

            module pm_copy = *src_pm;
            auto* new_pm =
                mpm.create_module(pm_name + ":split" + std::to_string(idx++), std::move(pm_copy));
            new_pm->set_bypass();

            m.replace_instruction(slice_ins, make_op("pointwise"), sliced_inputs, {new_pm});
        }

        changed = true;
    }

    return changed;
}

static bool find_pointwise_modules(module_pass_manager& mpm, bool multi_out)
{
    bool changed = false;
    auto last    = std::prev(mpm.get_module().end());
    for(auto ins : iterator_for(mpm.get_module()))
    {
        if(ins != last and is_dead(ins))
            continue;
        auto pw_outs = find_output_pointwise(mpm.get_module(), ins, multi_out);

        if(pw_outs.size() > 1)
        {
            (void)std::accumulate(
                pw_outs.begin() + 1, pw_outs.end(), pw_outs.front(), [&](auto input, auto output) {
                    return merge_instruction(mpm, input, output);
                });
            changed = true;
        }
        else if(ins->name() == "pointwise")
        {
            auto it = find_input_pointwise(mpm.get_module(), ins, multi_out);
            if(it == ins->inputs().end())
                continue;
            auto input = *it;
            if(is_dead(input))
                continue;
            merge_instruction(mpm, input, ins);

            changed = true;
        }
    }
    return changed;
}

namespace {
struct pointwise_reshape : rewrite_reshapes_base
{
    static std::string name() { return "pointwise"; }
};

} // namespace

void fuse_pointwise::apply(module_pass_manager& mpm) const
{
    mpm.run_pass(eliminate_identity{});
    create_pointwise_modules(mpm);
    mpm.run_pass(dead_code_elimination{});
    if(enabled(MIGRAPHX_DISABLE_POINTWISE_FUSION{}))
    {
        return;
    }
    for(int i = 0; i < 8; i++)
    {
        if(enable_rewrite_reshapes)
            mpm.run_pass(rewrite_reshapes<pointwise_reshape>{});
        if(enable_rewrite_broadcasts)
            rewrite_broadcasts(mpm, "pointwise");
        dedup_pointwise_inputs(mpm);
        auto changed = split_pointwise_through_slices(mpm);
        changed      = find_pointwise_modules(mpm, enable_multi_output) or changed;
        if(not changed)
            break;
        mpm.run_pass(dead_code_elimination{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
