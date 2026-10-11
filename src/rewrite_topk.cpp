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
 *
 */
#include <migraphx/rewrite_topk.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/array.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/split_factor.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

std::size_t get_tuple_index(instruction_ref elem)
{
    return elem->get_operator().to_value().at("index").to<std::size_t>();
}

/// Folds the converts of the indices into the topk, which then selects the
/// indices in that type directly
struct find_topk_convert
{
    auto matcher() const
    {
        return match::name("convert")(
            match::args(match::name("get_tuple_elem")(match::args(match::name("topk").bind("topk")))
                            .bind("indices")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto convert = r.result;
        auto indices = r.instructions["indices"];
        auto topk    = r.instructions["topk"];
        if(get_tuple_index(indices) != 1)
            return;
        // An index input fixes the type of the indices
        if(topk->inputs().size() != 1)
            return;
        auto type = convert->get_shape().type();
        // Every reader of the indices must want the same type
        if(not all_of(indices->outputs(), [&](instruction_ref out) {
               return out->name() == "convert" and out->get_shape().type() == type;
           }))
            return;
        if(indices->get_shape().type() != type)
        {
            auto op            = topk->get_operator().to_value();
            op["indices_type"] = type;
            auto new_topk      = m.insert_instruction(topk, make_op("topk", op), topk->inputs());
            m.replace_instruction(topk, new_topk);
        }
        auto converts = indices->outputs();
        std::for_each(converts.begin(), converts.end(), [&](instruction_ref out) {
            m.replace_instruction(out, indices);
        });
    }
};

struct find_large_topk
{
    std::size_t n_threshold = 0;
    auto matcher() const { return match::name("topk")(match::arg(0)(match::not_dynamic_shape())); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto input = ins->inputs().front();
        auto op    = ins->get_operator().to_value();
        auto axis  = op["axis"].to<std::int64_t>();
        auto dims  = input->get_shape().lens();
        auto n     = dims.at(axis);
        auto k     = op["k"].to<std::int64_t>();
        if(n < n_threshold)
            return;

        auto gdims = dims;
        // We have to sort at least k elements, so the min size is k*4 or half the threshold
        auto group = split_dim(gdims[axis], std::max<std::size_t>(n_threshold / 2, k * 4));
        if(group < 2)
            return;
        gdims.insert(gdims.begin() + axis, group);
        op["axis"] = axis + 1;

        // Both topk return the indices in the literal type, so the elements of
        // the final topk are converted back to the original type
        auto indices_type  = n < 65536 ? shape::uint16_type : shape::uint32_type;
        auto type          = ins->get_shape().sub_shapes().back().type();
        bool convert_elems = indices_type != type;
        if(convert_elems and not all_of(ins->outputs(), [](instruction_ref out) {
               return out->name() == "get_tuple_elem";
           }))
            return;

        auto fdims        = dims;
        fdims[axis]       = k * group;
        auto insert_final = [&](auto t, auto i) {
            auto elem = m.insert_instruction(ins, make_op("get_tuple_elem", {{"index", i}}), t);
            return m.insert_instruction(ins, make_op("reshape", {{"dims", fdims}}), elem);
        };

        std::vector<std::size_t> indices_data(n);
        std::iota(indices_data.begin(), indices_data.end(), 0);
        auto indices_lit = m.add_literal(shape{indices_type, {n}}, indices_data);

        auto indices = m.insert_instruction(
            ins, make_op("broadcast", {{"axis", axis}, {"out_lens", dims}}), indices_lit);
        auto gindices = m.insert_instruction(ins, make_op("reshape", {{"dims", gdims}}), indices);
        auto ginput   = m.insert_instruction(ins, make_op("reshape", {{"dims", gdims}}), input);
        auto topk1    = m.insert_instruction(ins, make_op("topk", op), ginput, gindices);
        auto finput   = insert_final(topk1, 0);
        auto findices = insert_final(topk1, 1);
        auto topk2    = m.insert_instruction(ins, ins->get_operator(), finput, findices);
        if(not convert_elems)
        {
            m.replace_instruction(ins, topk2);
            return;
        }
        auto elems = ins->outputs();
        std::for_each(elems.begin(), elems.end(), [&](instruction_ref elem) {
            auto new_elem = m.insert_instruction(elem, elem->get_operator(), topk2);
            if(get_tuple_index(elem) == 1)
                new_elem = m.insert_instruction(
                    elem, make_op("convert", {{"target_type", type}}), new_elem);
            m.replace_instruction(elem, new_elem);
        });
    }
};

} // namespace

void rewrite_topk::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm.get_module(), find_topk_convert{});
    // The split must not see the topk left behind by the folded converts
    mpm.run_pass(dead_code_elimination{});
    match::find_matches(mpm.get_module(), find_large_topk{split_threshold});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
