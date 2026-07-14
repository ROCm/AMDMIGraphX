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
#include <migraphx/gpu/rewrite_ssd.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/common.hpp>
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

/// Rewrite the dynamic nonmaxsuppression (NMS) slice feeding a topk into a static, mask-based form,
/// reduce the topk to the maximum k the model keeps, and move the num_selected trim to the outputs.
/// Common pattern seen in the SSD post-processing tail.
///
/// From:
///     topk(gather(scores, ...slice(get_tuple_elem[0](nms), get_tuple_elem[1](nms))...))
///     ... slice(get_tuple_elem[i](topk), k = min(cap, dim)) -> gathers -> outputs
/// To:
///     mask = less([0, 1, ..., max_boxes - 1], num_selected)
///     topk[k = cap](where(mask, gather(scores, ...get_tuple_elem[0](nms)...), sentinel))
///     ... gathers (static) ... slice(output, min(cap, num_selected))
struct find_nms_gather_topk : match::supports_dynamic_shapes
{
    auto matcher() const
    {
        return match::name("topk")(match::arg(0)(match::name("gather").bind("gather")));
    }

    // True when `ins` is `get_tuple_elem[index]` fed directly by a nonmaxsuppression.
    static bool is_nms_tuple_elem(instruction_ref ins, std::size_t index)
    {
        return ins->name() == "get_tuple_elem" and
               ins->get_operator().to_value().at("index").to<std::size_t>() == index and
               ins->inputs().at(0)->name() == "nonmaxsuppression";
    }

    // The variable-end `slice(get_tuple_elem[0](nms), get_tuple_elem[1](nms))` the NMS parser emits
    // to trim the padded indices down to num_selected.
    static bool is_nms_slice(instruction_ref ins)
    {
        if(ins->name() != "slice" or ins->inputs().size() != 2)
            return false;
        auto data = ins->inputs().at(0);
        auto ends = ins->inputs().at(1);
        return is_nms_tuple_elem(data, 0) and is_nms_tuple_elem(ends, 1) and
               data->inputs().at(0) == ends->inputs().at(0);
    }

    // Backward search over the inputs of `start` for the dynamic NMS slice, treating each as
    // terminal (its inputs are not traversed). Returns every one reached so the caller can require
    // that exactly one drives this gather.
    static std::vector<instruction_ref> find_nms_slices(instruction_ref start)
    {
        std::unordered_set<instruction_ref> seen;
        std::vector<instruction_ref> nms_slices;
        fix([&](auto self, instruction_ref ins) {
            if(not seen.insert(ins).second)
                return;
            if(is_nms_slice(ins))
            {
                nms_slices.push_back(ins); // terminal: do not traverse its inputs
                return;
            }
            for(auto input : ins->inputs())
                self(input);
        })(start);
        return nms_slices;
    }

    // A `slice(get_tuple_elem[i](topk), k)`: the variable-end slice parse_topk emits for a var-k.
    static bool is_topk_var_slice(instruction_ref ins, instruction_ref topk_ins)
    {
        if(ins->name() != "slice" or ins->inputs().size() != 2)
            return false;
        auto gte = ins->inputs().at(0);
        return gte->name() == "get_tuple_elem" and gte->inputs().at(0) == topk_ins;
    }

    static std::vector<instruction_ref> find_topk_var_slices(instruction_ref topk_ins)
    {
        std::vector<instruction_ref> slices;
        for(auto gte : topk_ins->outputs())
            for(auto out : gte->outputs())
                if(is_topk_var_slice(out, topk_ins))
                    slices.push_back(out);
        return slices;
    }

    // Backward search from `start` for the k-calc `concat`, returning its literal operands (the
    // maximum k the model keeps is the literal concatenated with the dim before the reduce_min).
    static std::vector<instruction_ref> find_cap_literals(instruction_ref start)
    {
        std::unordered_set<instruction_ref> seen;
        std::vector<instruction_ref> caps;
        fix([&](auto self, instruction_ref ins) {
            if(not seen.insert(ins).second)
                return;
            if(ins->name() == "concat")
            {
                for(auto input : ins->inputs())
                    if(input->name() == "@literal")
                        caps.push_back(input);
                return; // terminal
            }
            for(auto input : ins->inputs())
                self(input);
        })(start);
        return caps;
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto topk_ins   = mr.result;
        auto gather_ins = mr.instructions["gather"];

        // Only rewrite when a single dynamic NMS slice drives this gather.
        auto nms_slices = find_nms_slices(gather_ins);
        if(nms_slices.size() != 1)
            return;
        auto slice_ins    = nms_slices.front();
        auto indices      = slice_ins->inputs().at(0); // static [max_boxes, 3] get_tuple_elem[0]
        auto num_selected = slice_ins->inputs().at(1); // [1] get_tuple_elem[1]
        auto nms_ins      = indices->inputs().at(0);
        // The mask relies on num_selected being a scalar count that broadcasts against the iota.
        assert(num_selected->get_shape().elements() == 1);

        // max_boxes is the static leading dimension of the NMS selected-indices output.
        auto max_boxes = nms_ins->get_shape().sub_shapes().at(0).lens().at(0);

        // Only the 1D gathered-scores case (SSD post-processing) is handled; the mask then lines up
        // with the topk axis directly. The gather is still dynamic here, so check its max length.
        const auto& pre_shape = gather_ins->get_shape();
        if(pre_shape.ndim() != 1 or pre_shape.max_lens().at(0) != max_boxes)
            return;

        // Bypass the dynamic slice so the padded indices flow straight through. replace_instruction
        // propagates the now-static shapes down the chain, so the gathered scores become static
        // [max_boxes].
        m.replace_instruction(slice_ins, indices);

        const auto& data_shape = gather_ins->get_shape();
        auto largest           = topk_ins->get_operator().to_value().at("largest").to<bool>();

        // literal counting up 0, 1, ..., max_boxes - 1
        std::vector<int64_t> iota_data(max_boxes);
        std::iota(iota_data.begin(), iota_data.end(), 0);
        auto iota_lit = m.add_literal(literal{shape{shape::int64_type, {max_boxes}}, iota_data});

        // mask[i] = i < num_selected (from NMS)
        auto mask = insert_common_op(m, topk_ins, make_op("less"), {iota_lit, num_selected});
        mask      = m.insert_instruction(
            topk_ins, make_op("convert", {{"target_type", shape::bool_type}}), mask);

        // sentinel value that can never win the topk
        instruction_ref sentinel;
        data_shape.visit_type([&](auto dt) {
            auto sentinel_val = largest ? dt.min() : dt.max();
            sentinel = m.add_literal(literal{shape{data_shape.type(), {1}}, {sentinel_val}});
        });
        auto sentinel_bc = m.insert_instruction(
            topk_ins, make_op("multibroadcast", {{"out_lens", data_shape.lens()}}), sentinel);

        auto masked =
            m.insert_instruction(topk_ins, make_op("where"), mask, gather_ins, sentinel_bc);

        auto topk_inputs  = topk_ins->inputs();
        topk_inputs.at(0) = masked;
        m.replace_instruction(topk_ins, topk_ins->get_operator(), topk_inputs);

        // ---- SSD outputs: reduce the topk to the model's max k and trim only at the outputs ----
        auto topk_slices = find_topk_var_slices(topk_ins);
        if(topk_slices.empty())
            return; // not the SSD output tail; leave the mask-only form

        auto k_old = topk_slices.front()->inputs().at(1);
        auto caps  = find_cap_literals(k_old);
        if(caps.empty())
            return;
        auto cap_lit   = caps.front();
        auto cap_value = cap_lit->eval().at<int64_t>();

        // The topk only ever keeps `cap` results, so sort just those (static, far cheaper than
        // max_boxes). topk clamps k to the axis length internally.
        auto axis = topk_ins->get_operator().to_value().at("axis").to<int64_t>();
        if(axis < 0)
            axis += topk_ins->inputs().at(0)->get_shape().ndim();
        m.replace_instruction(
            topk_ins,
            make_op("topk", {{"k", cap_value}, {"axis", axis}, {"largest", largest}}),
            topk_ins->inputs());
        auto k_out = topk_ins->get_shape().sub_shapes().at(0).lens().at(axis);

        // Bypass parse_topk's variable-end slices so the gathers consume the full static topk
        // output and stay static.
        for(auto s : topk_slices)
            m.replace_instruction(s, s->inputs().at(0));

        auto ret = std::prev(m.end());
        if(ret->name() != "@return")
            return;

        // The only dynamic op(s): trim each topk-derived output to min(cap, num_selected).
        auto new_k = insert_common_op(m, ret, make_op("min"), {cap_lit, num_selected});

        std::unordered_set<instruction_ref> reachable;
        fix([&](auto self, instruction_ref ins) {
            if(not reachable.insert(ins).second)
                return;
            for(auto out : ins->outputs())
                self(out);
        })(topk_ins);

        auto ret_args = ret->inputs();
        for(auto arg : ret_args)
        {
            if(reachable.count(arg) == 0)
                continue;
            const auto& lens = arg->get_shape().lens();
            if(std::count(lens.begin(), lens.end(), k_out) != 1)
                continue;
            int64_t out_axis = std::find(lens.begin(), lens.end(), k_out) - lens.begin();
            auto trimmed     = m.insert_instruction(
                ret, make_op("slice", {{"starts", {0}}, {"axes", {out_axis}}}), arg, new_k);
            m.replace_instruction(arg, trimmed);
        }
    }
};

} // namespace

void rewrite_ssd::apply(module& m) const { match::find_matches(m, find_nms_gather_topk{}); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
