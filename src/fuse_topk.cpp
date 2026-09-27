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
#include <migraphx/fuse_topk.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/param_utils.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <iterator>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

bool is_topk(const instruction& ins) { return ins.name() == "topk"; }

/// A fused_reduce that already selects a topk
MIGRAPHX_PRED_MATCHER(fused_topk, instruction_ref ins)
{
    if(ins->name() != "fused_reduce")
        return false;
    return any_of(*ins->module_inputs().front(), &is_topk);
}

std::vector<std::int64_t> get_axes(instruction_ref reduce)
{
    return reduce->get_operator().to_value().at("axes").to_vector<std::int64_t>();
}

/// The axes a topk or a topk-carrying fused_reduce selects along
std::vector<std::int64_t> get_topk_axes(instruction_ref ins)
{
    if(ins->name() == "fused_reduce")
        return get_axes(ins);
    return {ins->get_operator().to_value().at("axis").to<std::int64_t>()};
}

std::size_t get_tuple_index(instruction_ref elem)
{
    return elem->get_operator().to_value().at("index").to<std::size_t>();
}

/// The topk can be selected in the workgroup of a reduction over the same
/// axis when it is small enough to sort in one workgroup
bool is_fusable_topk(instruction_ref topk,
                     const std::vector<std::int64_t>& axes,
                     std::size_t max_size)
{
    // The topk split by rewrite_topk carries an index input
    if(topk->inputs().size() != 1)
        return false;
    const auto& input = topk->inputs().front()->get_shape();
    if(input.dynamic())
        return false;
    if(get_topk_axes(topk) != axes)
        return false;
    if(axes.front() < 0)
        return false;
    std::size_t axis = axes.front();
    if(axis >= input.ndim())
        return false;
    return input.lens()[axis] <= max_size;
}

/// The fused_reduce output layout follows its inputs, so the fused outputs
/// must keep the shapes of the instructions they replace
bool same_output_shapes(const std::vector<instruction_ref>& inputs,
                        const std::vector<shape>& outputs)
{
    auto perm = find_permutation(to_shapes(inputs));
    return std::all_of(outputs.begin(), outputs.end(), [&](const shape& s) {
        return shape::from_permutation(s.type(), s.lens(), perm) == s;
    });
}

/// Whether ins is defined before pos in the module
bool is_before(module& m, instruction_ref ins, instruction_ref pos)
{
    auto r   = iterator_for(m);
    auto end = std::find(r.begin(), r.end(), pos);
    return std::find(r.begin(), end, ins) != end;
}

std::vector<instruction_ref>
insert_module_in_submodule(module_ref sm,
                           instruction_ref ins,
                           std::unordered_map<instruction_ref, instruction_ref>* map_ins)
{
    return sm->fuse(*ins->module_inputs().front(), ins->inputs(), map_ins);
}

/// Copies the topk or the topk-carrying fused_reduce into the submodule,
/// returning its tuple elements
std::vector<instruction_ref>
insert_topk_in_submodule(module_ref sm,
                         instruction_ref topk,
                         std::unordered_map<instruction_ref, instruction_ref>* map_ins)
{
    if(topk->name() == "fused_reduce")
        return insert_module_in_submodule(sm, topk, map_ins);
    auto t = sm->fuse({topk}, map_ins).front();
    std::vector<instruction_ref> result;
    auto is = range(topk->get_shape().sub_shapes().size());
    std::transform(is.begin(), is.end(), std::back_inserter(result), [&](auto i) {
        return sm->add_instruction(make_op("get_tuple_elem", {{"index", i}}), t);
    });
    return result;
}

void finalize_module(module_ref m)
{
    eliminate_common_subexpression{}.apply(*m);
    dead_code_elimination{}.apply(*m);
}

/// fused_reduce -> topk: the topk selects from the reduction output
struct find_reduce_topk
{
    std::size_t max_size = 0;

    auto matcher() const
    {
        return match::name("topk")(
            match::args(match::name("fused_reduce")(match::used_once()).bind("reduce")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto topk   = r.result;
        auto reduce = r.instructions["reduce"];
        if(not is_fusable_topk(topk, get_axes(reduce), max_size))
            return;
        if(not same_output_shapes(reduce->inputs(), topk->get_shape().sub_shapes()))
            return;

        const auto* old_rm = reduce->module_inputs().front();
        auto* rm           = mpm.create_module(old_rm->name() + ":topk");
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        map_ins[reduce] = insert_module_in_submodule(rm, reduce, &map_ins).front();
        rm->add_return(insert_topk_in_submodule(rm, topk, &map_ins));
        finalize_module(rm);

        auto new_inputs = find_inputs(map_ins, &mpm.get_module(), rm);
        mpm.get_module().replace_instruction(topk, reduce->get_operator(), new_inputs, {rm});
    }
};

/// topk -> fused_reduce: the reduction reads the topk elements. The elements
/// still read elsewhere become outputs of the fusion.
struct find_topk_reduce
{
    std::size_t max_size = 0;

    auto matcher() const
    {
        auto topk = match::any_of(match::name("topk"), fused_topk()).bind("topk");
        auto elem = match::name("get_tuple_elem")(match::used_once(), match::args(topk));
        return match::name("fused_reduce")(match::any_of[match::inputs()](elem));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto reduce = r.result;
        auto topk   = r.instructions["topk"];
        auto& m     = mpm.get_module();
        if(reduce->get_shape().type() == shape::tuple_type)
            return;
        auto axes = get_axes(reduce);
        if(get_topk_axes(topk) != axes)
            return;
        if(topk->name() == "topk" and not is_fusable_topk(topk, axes, max_size))
            return;
        if(not all_of(topk->outputs(),
                      [](instruction_ref out) { return out->name() == "get_tuple_elem"; }))
            return;

        // The elements read by the reduce are consumed by the fusion, so its
        // other inputs must be available where the topk is
        std::vector<instruction_ref> elems;
        std::vector<instruction_ref> others;
        std::partition_copy(reduce->inputs().begin(),
                            reduce->inputs().end(),
                            std::back_inserter(elems),
                            std::back_inserter(others),
                            [&](instruction_ref input) {
                                return input->name() == "get_tuple_elem" and
                                       input->inputs().front() == topk;
                            });
        if(not all_of(elems, [](instruction_ref elem) { return elem->outputs().size() == 1; }))
            return;
        if(not all_of(others, [&](instruction_ref input) { return is_before(m, input, topk); }))
            return;
        std::vector<instruction_ref> remaining;
        std::copy_if(topk->outputs().begin(),
                     topk->outputs().end(),
                     std::back_inserter(remaining),
                     [&](instruction_ref out) {
                         return not out->outputs().empty() and not contains(elems, out);
                     });

        auto inputs = topk->inputs();
        inputs.insert(inputs.end(), others.begin(), others.end());
        std::vector<shape> output_shapes = {reduce->get_shape()};
        std::transform(remaining.begin(),
                       remaining.end(),
                       std::back_inserter(output_shapes),
                       [](instruction_ref out) { return out->get_shape(); });
        if(not same_output_shapes(inputs, output_shapes))
            return;

        std::string prefix = topk->name() == "fused_reduce" ? topk->module_inputs().front()->name()
                                                            : m.name() + ":topk";
        auto* rm = mpm.create_module(prefix + ":" + reduce->module_inputs().front()->name());
        rm->set_bypass();
        std::unordered_map<instruction_ref, instruction_ref> map_ins;
        auto touts = insert_topk_in_submodule(rm, topk, &map_ins);
        std::transform(elems.begin(),
                       elems.end(),
                       std::inserter(map_ins, map_ins.end()),
                       [&](instruction_ref elem) {
                           return std::make_pair(elem, touts.at(get_tuple_index(elem)));
                       });
        auto returns = insert_module_in_submodule(rm, reduce, &map_ins);
        std::transform(remaining.begin(),
                       remaining.end(),
                       std::back_inserter(returns),
                       [&](instruction_ref out) { return touts.at(get_tuple_index(out)); });
        rm->add_return(returns);
        finalize_module(rm);

        auto new_inputs = find_inputs(map_ins, &m, rm);
        auto fused      = m.insert_instruction(topk, reduce->get_operator(), new_inputs, {rm});
        m.replace_instruction(reduce, make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto indices = range(std::size_t{1}, remaining.size() + 1);
        for_each(remaining.begin(),
                 remaining.end(),
                 indices.begin(),
                 [&](instruction_ref out, auto index) {
                     m.replace_instruction(
                         out, make_op("get_tuple_elem", {{"index", index}}), fused);
                 });
    }
};

} // namespace

void fuse_topk::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_reduce_topk{max_size});
    match::find_matches(mpm, find_topk_reduce{max_size});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
