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
#include <algorithm>
#include <cassert>
#include <iterator>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <unordered_set>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/shape_transform_descriptor.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/par.hpp>

#include <array>
#include <map>
#include <numeric>
#include <set>
#include <limits>
#include <variant>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

template <class Dims>
instruction_ref
insert_auto_reshape(module& m, instruction_ref ins, const Dims& dims, instruction_ref input)
{
    assert(std::all_of(dims.begin(), dims.end(), [](auto i) { return i > 0; }));
    if(std::equal(dims.begin(),
                  dims.end(),
                  input->get_shape().lens().begin(),
                  input->get_shape().lens().end()))
    {
        return input;
    }

    auto curr_lens = input->get_shape().lens();
    // Check if we can use squeeze (removing dimensions of size 1)
    if(curr_lens.size() > dims.size())
    {
        // Potential squeeze - check if we're just removing 1s
        std::vector<int64_t> axes_to_squeeze;
        std::size_t target_idx = 0;
        for(std::size_t curr_idx = 0; curr_idx < curr_lens.size(); ++curr_idx)
        {
            if(curr_lens[curr_idx] == 1)
            {
                axes_to_squeeze.push_back(curr_idx);
            }
            else
            {
                if(target_idx >= dims.size() or curr_lens[curr_idx] != dims[target_idx])
                {
                    axes_to_squeeze.clear();
                    break;
                }
                ++target_idx;
            }
        }
        if(not axes_to_squeeze.empty() and target_idx == dims.size())
        {
            return m.insert_instruction(
                ins, make_op("squeeze", {{"axes", axes_to_squeeze}}), input);
        }
    }
    // Check if we can use unsqueeze (adding dimensions of size 1)
    else if(curr_lens.size() < dims.size())
    {
        // Potential unsqueeze - check if we're just adding 1s
        std::vector<int64_t> axes_to_unsqueeze;
        std::size_t curr_idx = 0;
        for(std::size_t target_idx = 0; target_idx < dims.size(); ++target_idx)
        {
            if(dims[target_idx] == 1)
            {
                axes_to_unsqueeze.push_back(target_idx);
            }
            else
            {
                if(curr_idx >= curr_lens.size() or dims[target_idx] != curr_lens[curr_idx])
                {
                    axes_to_unsqueeze.clear();
                    break;
                }
                ++curr_idx;
            }
        }
        if(not axes_to_unsqueeze.empty() and curr_idx == curr_lens.size())
        {
            return m.insert_instruction(
                ins, make_op("unsqueeze", {{"axes", axes_to_unsqueeze}}), input);
        }
    }

    return m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), input);
}

template <class T>
instruction_ref insert_auto_reshape(module& m,
                                    instruction_ref ins,
                                    const std::initializer_list<T>& dims,
                                    instruction_ref input)
{
    return insert_auto_reshape(m, ins, std::vector<T>(dims), input);
}

const auto& reshaper_names()
{
    // clang-format off
    static const std::unordered_set<std::string> names = {
        "flatten",
        "reshape",
        "contiguous",
        "squeeze",
        "unsqueeze"
    };
    // clang-format on
    return names;
}

instruction_ref
insert_ops(module& m, instruction_ref ins, std::vector<operation>& ops, instruction_ref input)
{
    for(const auto& op : ops)
    {
        input = m.insert_instruction(ins, op, input);
    }
    return input;
}

struct find_nested_shape_transforms
{
    static const auto& shape_transform_ops()
    {
        static const std::unordered_set<std::string> names = {
            "flatten",
            "reshape",
            "squeeze",
            "unsqueeze",
            "transpose",
            "broadcast",
            "multibroadcast",
        };
        return names;
    }
    auto matcher() const
    {
        auto shape_transform = match::name(shape_transform_ops());
        auto output_not_shape_transform =
            match::none_of(match::skip_output(match::name("contiguous"))(shape_transform));
        auto input_has_shape_transform =
            match::args(match::skip(match::name("contiguous"))(shape_transform));
        return shape_transform(output_not_shape_transform, input_has_shape_transform);
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;

        std::vector<operation> ops;
        auto x = ins;
        while(contains(shape_transform_ops(), x->get_operator().name()) or
              x->get_operator().name() == "contiguous")
        {
            ops.push_back(x->get_operator());
            x = x->inputs().front();
        }
        if(x->get_shape().scalar())
        {
            m.replace_instruction(
                ins, make_op("multibroadcast", {{"out_lens", ins->get_shape().lens()}}), x);
        }
        else if(x->get_shape().elements() == 1 and ins->get_shape().elements() == 1)
        {
            // TODO: Use squeeze or unsqueeze
            m.replace_instruction(ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}), x);
        }
        else
        {
            std::reverse(ops.begin(), ops.end());
            auto opt_ops = optimize_shape_transforms(x->get_shape().lens(), ops);
            if(ops == opt_ops)
                return;
            auto y = insert_ops(m, ins, opt_ops, x);
            m.replace_instruction(ins, y);
        }
    }
};

struct find_op_shape_transform_op
{
    bool enable = true;

    static const auto& shape_transform_ops()
    {
        static const std::unordered_set<std::string> names = {
            "reshape",
            "squeeze",
            "unsqueeze",
            "flatten",
            "transpose",
            "contiguous",
            "multibroadcast",
            "broadcast",
        };
        return names;
    }

    static auto fusable_split()
    {
        return match::make_basic_pred_matcher([&](instruction_ref ins) {
            return any_of(ins->inputs(), [&](instruction_ref input_slice) {
                if(input_slice->name() != "slice")
                    return false;
                return all_of(input_slice->inputs().front()->outputs(), [&](instruction_ref slice) {
                    if(slice->name() != "slice")
                        return true;
                    return any_of(slice->outputs(),
                                  [&](instruction_ref x) { return x->name() == ins->name(); });
                });
            });
        });
    }

    auto matcher() const
    {
        auto reshapes = match::name(shape_transform_ops());
        auto match_op = match::any_of(match::reduce(), match::pointwise());
        auto x_op =
            match_op(match::none_of(fusable_split()));
        auto reshapes_x_op = reshapes(match::arg(0)(match::skip(reshapes())(x_op.bind("x"))));
        return match_op(match::any_of[match::inputs()](reshapes_x_op.bind("input")));
    }

    static bool matches_op(instruction_ref ins)
    {
        return is_reduce(ins) or ins->get_operator().attributes().contains("pointwise");
    }

    static bool is_reduce(instruction_ref ins) { return starts_with(ins->name(), "reduce_"); }

    template <class F>
    static instruction_ref find_input_if(instruction_ref start, instruction_ref last, F f)
    {
        while(start != last)
        {
            if(f(start))
                return start;
            if(start->inputs().size() != 1)
                return last;
            start = start->inputs().front();
        }
        return last;
    }

    template <class F>
    static bool any_input_of(instruction_ref start, instruction_ref last, F f)
    {
        return find_input_if(start, last, f) != last;
    }

    template <class AxesMap>
    static instruction_ref insert(module& m,
                                  instruction_ref ins,
                                  const std::vector<instruction_ref>& inputs,
                                  const AxesMap& am)
    {
        if(is_reduce(ins))
        {
            auto v       = ins->get_operator().to_value();
            auto op_axes = v.at("axes").to_vector<std::size_t>();
            std::vector<int64_t> axes;
            for(auto axis : op_axes)
            {
                auto new_axes = am.at(axis);
                axes.insert(axes.end(), new_axes.begin(), new_axes.end());
            }
            std::sort(axes.begin(), axes.end());
            v["axes"] = axes;
            return m.insert_instruction(ins, make_op(ins->name(), v), inputs, ins->module_inputs());
        }
        if(ins->name() == "layout")
        {
            auto v              = ins->get_operator().to_value();
            auto op_permutation = v.at("permutation").to_vector<std::int64_t>();
            std::vector<int64_t> permutation;
            for(auto axis : op_permutation)
            {
                auto new_axes = am.at(axis);
                permutation.insert(permutation.end(), new_axes.begin(), new_axes.end());
            }
            v["permutation"] = permutation;
            return m.insert_instruction(ins, make_op(ins->name(), v), inputs, ins->module_inputs());
        }
        return m.insert_instruction(ins, ins->get_operator(), inputs, ins->module_inputs());
    }

    static bool is_valid(instruction_ref ins, const shape_transform_descriptor& desc)
    {
        if(is_reduce(ins))
        {
            auto v       = ins->get_operator().to_value();
            auto op_axes = v.at("axes").to_vector<std::size_t>();
            std::sort(op_axes.begin(), op_axes.end());
            auto broadcasted_axes = desc.find_broadcasted_axes();
            return equal(op_axes, broadcasted_axes);
        }
        return desc.elements() == ins->get_shape().elements();
    }

    static std::vector<operation> generate(const shape_transform_descriptor& desc,
                                           const shape& input_shape)
    {
        if(input_shape.scalar() and input_shape.elements() == 1 and input_shape.ndim() == 1)
        {
            return {make_op("multibroadcast", {{"out_lens", desc.lens()}})};
        }
        else
        {
            return desc.generate(input_shape.lens());
        }
    }

    static shape_transform_descriptor
    make_descriptor(instruction_ref x_ins, std::vector<operation> ops, instruction_ref input_ins)
    {
        auto xinput            = x_ins->inputs().front();
        const auto& xlens      = x_ins->get_shape().lens();
        const auto& xinputlens = xinput->get_shape().lens();

        auto desc1 = shape_transform_descriptor::create(xlens, ops);
        if(not is_reduce(x_ins))
            return desc1;

        auto desc = desc1.rebase(xinputlens, true);
        if(not desc.empty())
            return desc;
        // We are broadcasting to a different size that doesnt match the input
        if(desc1.elements() != xinput->get_shape().elements() and
           desc1.elements() != x_ins->get_shape().elements())
        {
            // If we cant rebase the desc correctly then bail
            auto desc2 = desc1.rebase(xinputlens);
            if(desc2.elements() != xinput->get_shape().elements())
                return {};
            return desc1;
        }
        // Find a broadcast to append to improve the reduction analysis
        auto output_path = get_output_path(input_ins);
        auto it = std::find_if(output_path.begin(), output_path.end(), [&](instruction_ref ins) {
            if(ins->get_shape().lens() != input_ins->get_shape().lens())
                return true;
            return contains({"multibroadcast", "broadcast"}, ins->name());
        });
        if(it == output_path.end())
            return {};
        if(not contains({"multibroadcast", "broadcast"}, (*it)->name()))
            return {};
        ops.push_back((*it)->get_operator());
        return shape_transform_descriptor::create(xlens, ops).rebase(xinputlens, true);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        if(not enable)
            return;
        auto ins       = r.result;
        auto x_ins     = r.instructions["x"];
        auto input_ins = r.instructions["input"];

        // shape_transform_descriptor doesnt handle scalars for now
        if(input_ins->get_shape().scalar() or x_ins->get_shape().scalar())
            return;

        // If its just a broadcast then skip
        if(not any_input_of(input_ins, x_ins, [](instruction_ref x) {
               return not contains({"multibroadcast", "broadcast", "contiguous"}, x->name());
           }))
            return;

        std::vector<operation> ops;
        auto next_ins = input_ins;
        while(next_ins != x_ins)
        {
            ops.push_back(next_ins->get_operator());
            next_ins = next_ins->inputs().front();
        }
        assert(next_ins == x_ins);
        std::reverse(ops.begin(), ops.end());

        auto desc = make_descriptor(x_ins, ops, input_ins);
        if(desc.empty())
            return;

        if(not is_valid(x_ins, desc))
            return;

        // If we already in the common dimension space then skip if there are other outputs to avoid
        // infinite loop
        if(ins->get_shape().ndim() == desc.common_rank() and
           std::any_of(x_ins->outputs().begin(), x_ins->outputs().end(), [&](instruction_ref out) {
               return matches_op(out);
           }))
        {
            return;
        }

        auto reshape_input = [&](const auto& ins_to_insert, const auto& gdesc) {
            return [&](auto input) {
                auto gops = generate(gdesc, input->get_shape());
                return std::accumulate(
                    gops.begin(), gops.end(), input, [&](auto start, const auto& op) {
                        return m.insert_instruction(ins_to_insert, op, start);
                    });
            };
        };
        auto x_inputs = x_ins->inputs();
        std::transform(x_inputs.begin(),
                       x_inputs.end(),
                       x_inputs.begin(),
                       reshape_input(x_ins, desc.to_common_from_src()));
        auto new_input_ins = insert(m, x_ins, x_inputs, desc.common_axes_map_from_src());
        auto new_x_ins     = reshape_input(x_ins, desc.to_src_from_common())(new_input_ins);
        if(new_input_ins->get_shape().elements() != input_ins->get_shape().elements())
        {
            auto cdims    = desc.common_dims();
            new_input_ins = m.insert_instruction(
                x_ins, make_op("multibroadcast", {{"out_lens", cdims}}), new_input_ins);
        }
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            if(input == input_ins)
                return new_input_ins;
            return reshape_input(ins, desc.to_common_from_dst())(input);
        });
        // Replace old x_ins just in case it is used more than once
        assert(x_ins->get_shape().lens() == new_x_ins->get_shape().lens());
        m.replace_instruction(x_ins, new_x_ins);
        // Replace final instruction
        auto pw   = insert(m, ins, inputs, desc.common_axes_map_from_dst());
        auto rins = reshape_input(ins, desc.to_dst_from_common())(pw);
        assert(ins->get_shape().lens() == rins->get_shape().lens());
        m.replace_instruction(ins, rins);
    }
};

struct find_slice_shape_transforms
{
    static const auto& shape_transform_ops()
    {
        static const std::unordered_set<std::string> names = {
            "reshape",
            "squeeze",
            "unsqueeze",
            "flatten",
            "transpose",
            "contiguous",
            "multibroadcast",
            "broadcast",
        };
        return names;
    }

    // auto matcher() const
    // {
    //     auto reshapes = match::name(shape_transform_ops());
    //     auto match_op = match::any_of(match::reduce(), match::pointwise());
    //     auto x_op =
    //         match_op(match::none_of(fusable_split()));
    //     auto reshapes_x_op = reshapes(match::arg(0)(match::skip(reshapes())(x_op.bind("x"))));
    //     return match_op(match::any_of[match::inputs()](reshapes_x_op.bind("input")));
    // }

    auto matcher() const
    {
        auto reshapes = match::name(shape_transform_ops());
        auto slice_op = match::name("slice")(match::arg(0)(match::used_once()));
        return reshapes(reshapes(match::none_of[match::outputs()](reshapes())),
                        match::arg(0)(match::skip(reshapes())(slice_op.bind("slice"))));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins      = mr.result;
        auto slice    = mr.instructions["slice"];
        auto slice_op = slice->get_operator().to_value();
        auto axes     = slice_op.at("axes").to_vector<std::size_t>();

        std::vector<operation> ops;
        auto x = ins;
        while(contains(shape_transform_ops(), x->get_operator().name()))
        {
            ops.push_back(x->get_operator());
            x = x->inputs().front();
        }
        if(x != slice)
            return;
        x = x->inputs().front();
        std::reverse(ops.begin(), ops.end());
        auto desc = shape_transform_descriptor::create(slice->get_shape().lens(), ops);

        // std::cout << "desc: " << desc << std::endl;

        std::vector<std::size_t> new_axes;
        std::transform(axes.begin(),
                       axes.end(),
                       join_back_inserter(new_axes),
                       [&](auto axis) -> std::vector<std::size_t> {
                           auto result = desc.get_dst_axes_from_src(axis);
                           if(result.size() != 1)
                               return {};
                           return result;
                       });

        // Optimizes shape transforms if the slice cant be optimized
        if(axes.size() != new_axes.size())
        {
            auto opt_ops = desc.generate();
            auto y       = insert_ops(m, ins, opt_ops, slice);
            m.replace_instruction(ins, y);
            return;
        }
        slice_op["axes"] = new_axes;

        auto new_desc = desc.rebase(slice->inputs().front()->get_shape().lens());
        if(new_desc.empty())
            return;
        new_desc.simplify();

        auto opt_ops = new_desc.generate();
        auto y       = insert_ops(m, ins, opt_ops, x);
        y = m.insert_instruction(ins, make_op("slice", slice_op), y);
        m.replace_instruction(ins, y);

        // auto opt_ops = optimize_shape_transforms(x->get_shape().lens(), ops);
        // if(ops == opt_ops)
        //     return;
        // auto y = x;
        // for(const auto& op : opt_ops)
        //     y = m.insert_instruction(ins, op, y);
        // m.replace_instruction(ins, y);
        // if(x->get_shape().scalar())
        // {
        //     m.replace_instruction(
        //         ins, make_op("multibroadcast", {{"out_lens", ins->get_shape().lens()}}), x);
        // }
        // else if(x->get_shape().elements() == 1 and ins->get_shape().elements() == 1)
        // {
        //     // TODO: Use squeeze or unsqueeze
        //     m.replace_instruction(ins, make_op("reshape", {{"dims", ins->get_shape().lens()}}),
        //     x);
        // }
        // else
        // {
        // }
    }
};

struct find_nop_reshapes
{
    auto matcher() const
    {
        auto reshapes = reshaper_names();
        reshapes.insert("as_shape");
        reshapes.insert("broadcast");
        reshapes.insert("concat");
        reshapes.insert("convert");
        reshapes.insert("multibroadcast");
        reshapes.insert("pad");
        reshapes.insert("slice");
        reshapes.insert("step");
        reshapes.insert("transpose");
        reshapes.insert("reduce_mean");
        reshapes.insert("reduce_max");
        reshapes.insert("reduce_min");
        reshapes.insert("reduce_sum");
        reshapes.insert("reduce_prod");
        return match::name(reshapes)(match::same_shape(match::arg(0)));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        m.replace_instruction(ins, ins->inputs().front());
    }
};

struct find_nested_slice
{
    auto matcher() const { return match::name("slice")(match::arg(0)(match::name("slice"))); }

    using axes_map = std::map<std::size_t, std::pair<std::size_t, std::size_t>>;

    static axes_map get_axes(instruction_ref ins)
    {
        axes_map result;
        auto op = any_cast<op::slice>(ins->get_operator());
        for(std::size_t i = 0; i < op.axes.size(); i++)
        {
            result[op.axes[i]] = std::make_pair(op.starts[i], op.ends[i]);
        }
        return result;
    }

    static axes_map merge(const axes_map& m1, const axes_map& m2)
    {
        axes_map result;
        // Non overlapping
        for(auto&& p : m1)
        {
            if(contains(m2, p.first))
                continue;
            result[p.first] = p.second;
        }
        for(auto&& p : m2)
        {
            if(contains(m1, p.first))
                continue;
            result[p.first] = p.second;
        }
        // Overlapping
        for(auto&& p1 : m1)
        {
            if(not contains(m2, p1.first))
                continue;
            auto&& v1        = p1.second;
            auto&& v2        = m2.at(p1.first);
            auto start       = v1.first + v2.first;
            auto end         = start + (v2.second - v2.first);
            result[p1.first] = std::make_pair(start, end);
        }
        return result;
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins   = mr.result;
        auto slice = ins->inputs().front();
        auto input = slice->inputs().front();

        auto a1 = get_axes(ins);
        auto a2 = get_axes(slice);

        auto axes = merge(a2, a1);

        auto op = op::slice{};
        for(auto&& pp : axes)
        {
            op.axes.push_back(pp.first);
            op.starts.push_back(pp.second.first);
            op.ends.push_back(pp.second.second);
        }
        m.replace_instruction(ins, op, input);
    }
};

/**
 *  Example case
 *  From:
 *  param0: lens = [3, 4], strides = [4, 1]
 *  param1: lens = [3, 4], strides = [4, 1]
 *  mb0: multibroadcast(param0, output_lens = [2, 3, 4])
 *  mb1: multibroadcast(param1, output_lens = [2, 3, 4])
 *  concat(mb0, mb1, axis = 2)
 *
 *  To:
 *  param0: lens = [3, 4], strides = [4, 1]
 *  param1: lens = [3, 4], strides = [4, 1]
 *  con0: concat(param0, param1, axis = 1)
 *  multibroadcast(con0, lens = [2, 3, 4])
 */
struct find_concat_multibroadcasts
{
    auto matcher() const
    {
        return match::name("concat")(
            match::all_of[match::inputs()](match::name("multibroadcast", "broadcast")));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto concat_ins       = mr.result;
        auto concat_op        = any_cast<op::concat>(concat_ins->get_operator());
        auto concat_out_lens  = concat_ins->get_shape().lens();
        auto concat_inputs    = concat_ins->inputs();
        auto front_mb_strides = concat_inputs.front()->get_shape().strides();
        assert(concat_op.axis >= 0);

        // Only apply when concat axis is not a broadcasted dimension
        if(std::any_of(concat_inputs.begin(), concat_inputs.end(), [&](auto i) {
               return i->get_shape().strides()[concat_op.axis] == 0;
           }))
        {
            return;
        }

        // Skip if the broadcasts are different
        auto broadcast       = concat_inputs.front()->get_operator();
        auto broadcast_value = broadcast.to_value();
        if(not std::all_of(concat_inputs.begin() + 1, concat_inputs.end(), [&](instruction_ref b) {
               if(b->name() != broadcast.name())
                   return false;
               if(broadcast.name() == "broadcast")
                   return b->get_operator().to_value()["axis"] == broadcast_value["axis"];
               return true;
           }))
        {
            return;
        }

        // Get the inputs of multibroadcast ops. Will be used as inputs to new concat op
        std::vector<instruction_ref> inputs(concat_inputs.size());
        std::transform(concat_inputs.begin(), concat_inputs.end(), inputs.begin(), [](auto i) {
            return i->inputs().front();
        });

        // Check that the inputs into the broadcasts have the same rank
        const auto& first_shape = inputs.front()->get_shape();
        if(not std::all_of(inputs.begin() + 1, inputs.end(), [&](auto input) {
               return input->get_shape().ndim() == first_shape.ndim();
           }))
        {
            return;
        }

        // Reduce axis by number of leading broadcasted dimensions
        if(inputs.front()->get_shape().lens().size() < concat_out_lens.size())
        {
            concat_op.axis -=
                std::count(front_mb_strides.begin(), front_mb_strides.begin() + concat_op.axis, 0);
        }

        // Inputs to broadcasts should have the same dimensions except for the axis to
        // concatenate over
        const auto& front_in_lens = inputs.front()->get_shape().lens();
        if(not std::all_of(inputs.begin() + 1, inputs.end(), [&](auto input_to_mb) {
               const auto& lens = input_to_mb->get_shape().lens();
               return std::equal(
                          lens.begin(), lens.begin() + concat_op.axis, front_in_lens.begin()) and
                      std::equal(lens.begin() + concat_op.axis + 1,
                                 lens.end(),
                                 front_in_lens.begin() + concat_op.axis + 1);
           }))
        {
            return;
        }

        auto new_concat_ins = m.insert_instruction(concat_ins, concat_op, inputs);
        broadcast.from_value({{"out_lens", concat_ins->get_shape().lens()}});
        m.replace_instruction(concat_ins, broadcast, new_concat_ins);
    }
};

struct find_concat_slice
{
    auto matcher() const
    {
        return match::name("concat")(match::any_of[match::outputs()](match::name("slice")));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins    = mr.result;
        auto inputs = ins->inputs();
        auto outs   = ins->outputs();
        std::vector<migraphx::instruction_ref> slice_ins;
        migraphx::transform_if(
            outs.begin(),
            outs.end(),
            std::back_inserter(slice_ins),
            [&](const auto& oins) { return oins->name() == "slice"; },
            [&](const auto& oins) { return oins; });
        int concat_axis = any_cast<op::concat>(ins->get_operator()).axis;
        // prune slice candidates
        std::vector<migraphx::instruction_ref> slice_candidates;
        for(const auto& sins : range(slice_ins.begin(), slice_ins.end()))
        {
            auto sop = any_cast<op::slice>(sins->get_operator());
            // slices with only one axis is allowed, because concat happens only one axis
            if(sop.axes.size() != 1 or sop.axes.front() != concat_axis)
            {
                continue;
            }
            slice_candidates.push_back(sins);
        }
        if(slice_candidates.empty())
        {
            return;
        }
        std::vector<size_t> prefix_scan = {0};
        std::transform(
            inputs.begin(), inputs.end(), std::back_inserter(prefix_scan), [&](const auto& i) {
                return prefix_scan.back() + i->get_shape().lens()[concat_axis];
            });
        for(const auto& sins : slice_candidates)
        {
            auto sop           = any_cast<op::slice>(sins->get_operator());
            size_t slice_start = sop.starts.front();
            size_t slice_len   = sop.ends.front() - slice_start;
            auto fii = std::find_if(prefix_scan.begin(), prefix_scan.end(), [&](const auto& j) {
                return j == slice_start;
            });
            if(fii == prefix_scan.end())
            {
                continue;
            }
            // slice_len == 0
            else if(fii == prefix_scan.end() - 1)
            {
                assert(slice_len == 0 or slice_start >= prefix_scan.back());
                continue;
            }
            else
            {
                size_t idx = std::distance(prefix_scan.begin(), fii);
                if(inputs[idx]->get_shape().lens()[concat_axis] == slice_len)
                {
                    assert((prefix_scan[idx + 1] - prefix_scan[idx]) == slice_len);
                    m.replace_instruction(sins, inputs[idx]);
                }
            }
        }
    }
};

struct find_concat_transpose
{
    auto matcher() const
    {
        return match::name("concat")(match::all_of[match::inputs()](match::name("transpose")));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins          = mr.result;
        auto trans_inputs = ins->inputs();
        auto s            = trans_inputs.front()->get_shape();
        assert(s.transposed());
        auto op          = any_cast<op::concat>(ins->get_operator());
        auto permutation = find_permutation(s);

        // permutation should be the same for all inputs
        if(not std::all_of(trans_inputs.begin(), trans_inputs.end(), [&](auto in) {
               return (find_permutation(in->get_shape()) == permutation);
           }))
        {
            return;
        }

        // axis could be a negative value
        int64_t n_dim = s.lens().size();
        op.axis       = tune_axis(n_dim, op.axis, op.name());

        auto ipermutation = invert_permutation(permutation);
        op.axis           = ipermutation[op.axis];

        std::vector<instruction_ref> inputs;
        std::transform(
            ins->inputs().begin(), ins->inputs().end(), std::back_inserter(inputs), [&](auto i) {
                return m.insert_instruction(
                    ins, make_op("transpose", {{"permutation", permutation}}), i);
            });
        auto concat = m.insert_instruction(ins, op, inputs);
        auto t      = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", ipermutation}}), concat);
        assert(ins->get_shape().lens() == t->get_shape().lens());
        m.replace_instruction(ins, t);
    }
};

struct find_concat_reshape
{
    auto matcher() const
    {
        return match::name("concat")(match::all_of[match::inputs()](
            match::name("reshape", "unsqueeze", "squeeze", "reshape_lazy")));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins          = mr.result;
        auto concat_shape = ins->get_shape();
        auto reshapes     = ins->inputs();
        if(reshapes.empty())
            return;
        auto input_shape = reshapes.front()->inputs().front()->get_shape();
        // All inputs should have the same dimensions
        if(not std::all_of(
               std::next(reshapes.begin()), reshapes.end(), [&](instruction_ref reshape) {
                   return reshape->inputs().front()->get_shape().lens() == input_shape.lens();
               }))
            return;
        // axis could be a negative value
        auto op       = any_cast<op::concat>(ins->get_operator());
        int64_t n_dim = reshapes.front()->get_shape().lens().size();
        auto axis     = tune_axis(n_dim, op.axis, op.name());

        auto predims  = std::accumulate(concat_shape.lens().begin(),
                                       concat_shape.lens().begin() + axis,
                                       std::size_t{1},
                                       std::multiplies<>{});
        auto postdims = std::accumulate(concat_shape.lens().begin() + axis + 1,
                                        concat_shape.lens().end(),
                                        std::size_t{1},
                                        std::multiplies<>{});

        // Find the axis on the input
        std::size_t x = 1;
        auto it = std::find_if(input_shape.lens().begin(), input_shape.lens().end(), [&](auto d) {
            x *= d;
            return x > predims;
        });
        if(it == input_shape.lens().end())
            return;
        op.axis       = it - input_shape.lens().begin();
        auto ipredims = std::accumulate(input_shape.lens().begin(),
                                        input_shape.lens().begin() + op.axis,
                                        std::size_t{1},
                                        std::multiplies<>{});
        if(ipredims != predims)
            return;
        auto ipostdims = std::accumulate(input_shape.lens().begin() + op.axis + 1,
                                         input_shape.lens().end(),
                                         std::size_t{1},
                                         std::multiplies<>{});
        if(ipostdims != postdims)
            return;

        std::vector<instruction_ref> inputs;
        std::transform(reshapes.begin(),
                       reshapes.end(),
                       std::back_inserter(inputs),
                       [&](instruction_ref i) { return i->inputs().front(); });
        auto concat = m.insert_instruction(ins, op, inputs);
        m.replace_instruction(ins, make_op("reshape", {{"dims", concat_shape.lens()}}), concat);
    }
};

struct find_nested_concat
{
    auto matcher() const
    {
        return match::name("concat")(match::any_of[match::inputs()](match::name("concat")));
    }

    static std::size_t get_axis(instruction_ref ins)
    {
        auto op = any_cast<op::concat>(ins->get_operator());
        return op.axis;
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins  = mr.result;
        auto axis = get_axis(ins);
        std::vector<instruction_ref> args;
        fix([&](auto self, auto&& inputs) {
            for(auto&& i : inputs)
            {
                if(i->name() == "concat" and get_axis(i) == axis and i->outputs().size() == 1)
                    self(i->inputs());
                else
                    args.push_back(i);
            }
        })(ins->inputs());
        m.replace_instruction(ins, ins->get_operator(), args);
    }
};

struct find_gather
{
    struct arithmetic_segment
    {
        int64_t base      = 0;
        int64_t stride    = 0;
        std::size_t count = 0;

        template <class Iterator>
        static std::vector<arithmetic_segment> from_ints(Iterator begin, Iterator end)
        {
            std::vector<arithmetic_segment> result(std::distance(begin, end));
            par_transform(
                begin, end, result.begin(), [](auto x) { return arithmetic_segment{x, 1, 1}; });
            return result;
        }

        template <class Iterator, class OutputIterator>
        static Iterator find_largest(Iterator start, Iterator last, OutputIterator out)
        {
            for(auto it = start; it != last;)
            {
                auto [seg, next_it] = find(it, last);
                it                  = next_it;
                *out                = seg;
                out++;
            }
            return last;
        }

        template <class Iterator, class OutputIterator>
        static Iterator find_n(Iterator start, Iterator last, std::size_t n, OutputIterator out)
        {
            for(auto it = start; it != last;)
            {
                // Check if we have at least n elements remaining
                auto remaining = static_cast<std::size_t>(std::distance(it, last));
                if(remaining < n)
                    return it; // Not enough elements for another segment
                auto [seg, next_it] = find(it, it + n);
                if(next_it != it + n)
                    return next_it;
                it   = next_it;
                *out = seg;
                out++;
            }
            return last;
        }

        static std::vector<arithmetic_segment>
        make_segments(const std::vector<arithmetic_segment>& segments, bool uniform = true)
        {
            std::vector<arithmetic_segment> result;
            auto [first_seg, first_it] = find(segments.begin(), segments.end());
            result.push_back(first_seg);
            // Try to find segments that are the same size
            auto it = find_n(first_it, segments.end(), first_seg.count, std::back_inserter(result));
            if(it != segments.end())
            {
                if(uniform)
                    return {};
                result.resize(1);
                find_largest(first_it, segments.end(), std::back_inserter(result));
            }
            return result;
        }

        static std::vector<arithmetic_segment> shift(std::vector<arithmetic_segment> segments,
                                                     std::int64_t shift)
        {
            par_transform(
                segments.begin(), segments.end(), segments.begin(), [&](arithmetic_segment x) {
                    x.base += shift;
                    return x;
                });
            return segments;
        }

        /// Detect arithmetic segment pattern
        template <class Iterator>
        static std::pair<arithmetic_segment, Iterator> find(Iterator begin, Iterator end)
        {
            std::size_t length = std::distance(begin, end);
            if(length == 0)
                return std::make_pair(arithmetic_segment{}, begin);
            if(length == 1)
                return std::make_pair(*begin, std::next(begin));
            auto start = *begin;
            // auto base   = *begin;
            auto stride = std::next(begin)->base - start.base;
            if(stride < 0)
                return std::make_pair(*begin, std::next(begin));
            auto diff =
                std::adjacent_find(begin, end, [&](arithmetic_segment x, arithmetic_segment y) {
                    return y.base - x.base != stride;
                });
            if(diff != end)
                diff++;
            return std::make_pair(
                arithmetic_segment{start.base, stride, std::size_t(std::distance(begin, diff))},
                diff);
        }

        static shape make_strided_view(std::vector<arithmetic_segment> segments)
        {
            std::vector<std::size_t> lens;
            std::vector<std::size_t> strides;

            do
            {
                segments = make_segments(segments);
                // std::cout << "nsegments: " << segments.size() << std::endl;
                // for(auto segment : segments)
                //     std::cout << "    {" << segment.base << ", " << segment.stride << ", "
                //               << segment.count << "}\n";
                if(segments.empty())
                    return {};
                auto seg = segments.front();
                if(seg.stride < 0)
                    return {};
                if(std::any_of(segments.begin(), segments.end(), [](const arithmetic_segment& seg) {
                       return seg.base < 0;
                   }))
                    return {};
                if(not std::all_of(
                       segments.begin(), segments.end(), [&](const arithmetic_segment& seg) {
                           return seg.stride == segments.front().stride and
                                  seg.count == segments.front().count;
                       }))
                    return {};
                lens.push_back(seg.count);
                strides.push_back(seg.stride);
            } while(segments.size() > 1);

            std::reverse(lens.begin(), lens.end());
            std::reverse(strides.begin(), strides.end());

            if(std::none_of(
                   strides.begin(), strides.end(), [](auto stride) { return stride == 1; }))
            {
                lens.push_back(1);
                strides.push_back(1);
            }

            return {shape::float_type, lens, strides};
        }

        // Find where segment pattern breaks and return the split point
        // Returns 0 if no valid split found, otherwise returns index where to split
        static std::size_t find_segment_split(const std::vector<arithmetic_segment>& segments)
        {
            if(segments.size() < 2)
                return 0;

            auto [first_seg, first_it] = find(segments.begin(), segments.end());
            if(first_seg.count == segments.size())
                return 0; // All segments form one pattern, no split needed

            // The split point is where the first segment pattern ends
            return first_seg.count;
        }

        // Recursively create strided views by splitting at pattern breaks
        // Returns a vector of (shape, offset) pairs for each view
        static std::vector<std::pair<shape, std::int64_t>>
        make_strided_views_recursive(const std::vector<arithmetic_segment>& segments,
                                     std::size_t depth,
                                     std::size_t max_views)
        {
            // Limit recursion depth to avoid pathological cases
            if(depth > 8 or segments.empty())
                return {};

            // First try to create a single strided view for all segments
            std::int64_t offset = segments.front().base;
            auto shifted        = shift(std::vector<arithmetic_segment>(segments), -offset);
            auto s              = make_strided_view(shifted);

            if(not s.lens().empty() and s.elements() > 0)
            {
                // Success - return this as a single view
                // Accept both standard and non-standard shapes
                return {{s, offset}};
            }

            // Find where the pattern breaks
            std::size_t split_point = find_segment_split(segments);
            if(split_point == 0 or split_point >= segments.size())
                return {};

            // Split segments at the break point
            std::vector<arithmetic_segment> first_part(segments.begin(),
                                                       segments.begin() + split_point);
            std::vector<arithmetic_segment> rest_part(segments.begin() + split_point, segments.end());

            // Recursively process each part
            auto first_views = make_strided_views_recursive(first_part, depth + 1, max_views);
            if(first_views.empty())
                return {};

            // Check if we've exceeded max views
            if(first_views.size() >= max_views)
                return {};

            auto rest_views =
                make_strided_views_recursive(rest_part, depth + 1, max_views - first_views.size());
            if(rest_views.empty())
                return {};

            // Combine results
            first_views.insert(first_views.end(), rest_views.begin(), rest_views.end());

            // Limit total number of views
            if(first_views.size() > max_views)
                return {};

            return first_views;
        }

        // Attempt to create multiple strided views that can be concatenated
        // Returns a vector of (shape, offset) pairs for each view
        static std::vector<std::pair<shape, std::int64_t>>
        make_strided_views(const std::vector<arithmetic_segment>& segments,
                          std::size_t input_elements)
        {
            // Allow more views for larger inputs, but cap at reasonable limit
            std::size_t max_views = std::min<std::size_t>(16, std::max<std::size_t>(4, input_elements / 16));
            auto views = make_strided_views_recursive(segments, 0, max_views);

            // Need at least 2 views for concat to make sense
            if(views.size() < 2)
                return {};


            return views;
        }

        template <class Indices>
        static std::optional<instruction_ref>
        transform_indices(const Indices& indices, module& m, instruction_ref start)
        {
            auto isegments      = from_ints(indices.begin(), indices.end());
            std::int64_t offset = isegments.front().base;
            auto shifted        = shift(isegments, -offset);
            auto s              = make_strided_view(shifted);
            std::size_t input_elements = start->get_shape().elements();
            auto ops = generate_shape_transforms_for(s, {input_elements}, offset);
            if(ops.has_value())
                return insert_ops(m, std::next(start), *ops, start);

            // Try to create multiple strided views that can be concatenated
            auto views = make_strided_views(isegments, input_elements);
            if(views.empty())
                return std::nullopt;

            return transform_indices_with_concat(views, m, start, input_elements);
        }

        static std::optional<instruction_ref>
        transform_indices_with_concat(const std::vector<std::pair<shape, std::int64_t>>& views,
                                      module& m,
                                      instruction_ref start,
                                      std::size_t input_elements)
        {
            std::vector<instruction_ref> concat_inputs;

            for(const auto& [vs, view_offset] : views)
            {
                std::size_t view_elements = vs.elements();
                instruction_ref view_ins;

                // For simple 1D contiguous views, use direct slice (simpler than generate_shape_transforms_for)
                if(vs.standard() and vs.lens().size() == 1 and vs.strides().front() == 1)
                {
                    view_ins = m.insert_instruction(
                        std::next(start),
                        make_op("slice",
                                {{"axes", {0}},
                                 {"starts", {view_offset}},
                                 {"ends", {view_offset + static_cast<std::int64_t>(view_elements)}}}),
                        start);
                }
                else
                {
                    // Use generate_shape_transforms_for for complex strided views
                    auto ops = generate_shape_transforms_for(vs, {input_elements}, view_offset);
                    if(not ops.has_value())
                        return std::nullopt;

                    view_ins = insert_ops(m, std::next(start), *ops, start);
                    
                    // Reshape to 1D for concat
                    view_ins = m.insert_instruction(
                        std::next(start), make_op("reshape", {{"dims", {view_elements}}}), view_ins);
                }
                concat_inputs.push_back(view_ins);
            }

            if(concat_inputs.size() < 2)
                return std::nullopt;

            // Concatenate along axis 0
            return m.insert_instruction(
                std::next(start), make_op("concat", {{"axis", 0}}), concat_inputs);
        }
    };

    static std::vector<std::int64_t> build_flat_gather_indices(instruction_ref gather_ins,
                                                               const argument& indices_arg,
                                                               std::size_t axis_index)
    {
        auto data_ins    = gather_ins->inputs()[0];
        auto output_dims = gather_ins->get_shape().lens();
        const auto r_in  = data_ins->get_shape().lens().size();
        const auto r_idx = indices_arg.get_shape().lens().size();
        assert(axis_index < r_in);

        shape output_s{shape::float_type, output_dims}; // element type doesn't matter here
        const auto out_n = output_s.elements();
        std::vector<std::int64_t> flat(out_n);
        std::iota(flat.begin(), flat.end(), 0);

        auto indices = indices_arg.to_vector<int64_t>();

        transform(flat, flat.begin(), [&](std::size_t out_lin) -> std::int64_t {
            // 1) output linear -> output multi-index
            auto out_multi = output_s.multi(out_lin);

            // 2) isolate the "indices" coordinates from the output coords (inserted at `axis`)
            std::vector<std::size_t> idx_multi(r_idx);
            std::copy(out_multi.begin() + axis_index,
                      out_multi.begin() + axis_index + r_idx,
                      idx_multi.begin());

            // 3) look up the actual index value (may be negative)
            const std::int64_t idx_lin  = indices_arg.get_shape().index(idx_multi);
            const std::int64_t axis_len = data_ins->get_shape().lens().at(axis_index);
            auto idx_val                = indices.at(idx_lin);

            // Normalize negative indices into [0, axis_len)
            if(idx_val < 0)
                idx_val += axis_len;

            assert(idx_val >= 0 and idx_val < axis_len);

            // 4) construct corresponding INPUT multi-index
            std::vector<std::size_t> in_multi(r_in);

            // copy dims before axis
            std::copy(out_multi.begin(), out_multi.begin() + axis_index, in_multi.begin());

            // axis coordinate from indices
            in_multi.at(axis_index) = idx_val;

            // copy dims after axis; they are shifted by r_idx in output
            std::copy(out_multi.begin() + axis_index + r_idx,
                      out_multi.end(),
                      in_multi.begin() + axis_index + 1);

            // 5) map input multi-index -> flat offset in contiguous buffer
            const auto in_lin = data_ins->get_shape().index(in_multi);
            return in_lin;
        });

        return flat;
    }
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::any().bind("data"), match::is_constant().bind("indices")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins          = r.result;
        auto indices_ins  = r.instructions["indices"];
        auto data_ins     = r.instructions["data"];
        auto gather_op    = any_cast<op::gather>(ins->get_operator());
        const auto& dlens = data_ins->get_shape().lens();
        if(dlens.empty())
            return;

        const auto axis_index = static_cast<std::size_t>(
            tune_axis(static_cast<int>(dlens.size()), gather_op.axis, gather_op.name()));
        const auto axis_len = dlens.at(axis_index);
        if(axis_len == 0)
            return;

        auto arg_ind = indices_ins->eval();
        if(arg_ind.empty())
            return;

        std::vector<std::int64_t> indices_values;
        arg_ind.visit([&](auto v) {
            indices_values.resize(v.size());
            std::transform(v.begin(), v.end(), indices_values.begin(), [](auto x) {
                return static_cast<std::int64_t>(x);
            });
        });
        if(indices_values.empty())
            return;

        const auto& indices_shape = indices_ins->get_shape();
        if(indices_shape.elements() != indices_values.size())
            return;

        // Skip if indices have broadcast strides (e.g., scalar broadcast)
        if(indices_shape.broadcasted())
            return;

        // Normalize negative indices using transform
        std::transform(indices_values.begin(),
                       indices_values.end(),
                       indices_values.begin(),
                       [axis_len](auto idx) {
                           if(idx < 0)
                               idx += static_cast<std::int64_t>(axis_len);
                           return idx;
                       });

        // Validate all indices are in bounds
        bool all_valid =
            std::all_of(indices_values.begin(), indices_values.end(), [axis_len](auto idx) {
                return idx >= 0 and idx < static_cast<std::int64_t>(axis_len);
            });
        if(not all_valid)
            return;

        // Create indices argument with normalized values
        shape normalized_indices_shape{shape::int64_type, indices_shape.lens()};
        literal indices_lit(normalized_indices_shape, indices_values.begin(), indices_values.end());
        argument indices_arg = indices_lit.get_argument();

        // Sanity check: ensure the argument shape matches
        assert(indices_arg.get_shape().lens() == indices_shape.lens());
        assert(indices_arg.get_shape().elements() == indices_values.size());

        std::optional<instruction_ref> new_ins = std::nullopt;

        if(data_ins->get_shape().ndim() == 1 and indices_ins->get_shape().ndim() == 1)
        {
            new_ins = arithmetic_segment::transform_indices(indices_values, m, data_ins);
        }
        else
        {
            auto data_1d =
                insert_auto_reshape(m, ins, {data_ins->get_shape().elements()}, data_ins);
            auto new_indices = build_flat_gather_indices(ins, indices_arg, axis_index);
            new_ins          = arithmetic_segment::transform_indices(new_indices, m, data_1d);
        }

        if(not new_ins.has_value())
            return;

        auto reshaped = insert_auto_reshape(m, ins, ins->get_shape().lens(), *new_ins);

        m.replace_instruction(ins, reshaped);
    }
};

struct find_reshape_cont
{
    auto matcher() const
    {
        auto contiguous = match::skip(match::name("contiguous"))(
            match::none_of(match::standard_shape()).bind("input"));
        auto reshape_contiguous = match::name("reshape")(match::args(contiguous));
        return match::pointwise(
            match::nargs(2), match::either_arg(0, 1)(reshape_contiguous.bind("rsp"), match::any()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto cont_input = r.instructions["input"];
        auto in_ins   = r.instructions["rsp"];

        auto lens       = cont_input->get_shape().lens();
        std::vector<int64_t> dims(lens.begin(), lens.end());

        if(in_ins->get_shape() != ins->get_shape())
        {
            return;
        }

        if(not std::all_of(ins->inputs().begin(), ins->inputs().end(), [](auto i) {
               return i->get_shape().standard();
           }))
        {
            return;
        }

        auto out_lens = ins->get_shape().lens();
        std::vector<int64_t> out_dims(out_lens.begin(), out_lens.end());
        std::vector<instruction_ref> inputs;
        for(const auto& in : ins->inputs())
        {
            if(in == in_ins)
            {
                inputs.push_back(cont_input);
            }
            else
            {
                inputs.push_back(
                    m.insert_instruction(ins, make_op("reshape", {{"dims", dims}}), in));
            }
        }
        auto out = m.insert_instruction(ins, ins->get_operator(), inputs);
        m.replace_instruction(ins, make_op("reshape", {{"dims", out_dims}}), out);
    }
};

struct find_unary_shape_transforms
{
    static const auto& shape_transforms()
    {
        static const std::unordered_set<std::string> names = {
            "flatten",
            "reshape",
            "squeeze",
            "unsqueeze",
            "transpose",
            "broadcast",
            "multibroadcast",
        };
        return names;
    }
    auto matcher() const
    {
        auto output_not_pointwise =
            match::none_of(match::skip_output(match::name("contiguous"))(match::pointwise()));
        auto shape_transform = match::name(shape_transforms());
        auto input_has_shape_transform =
            match::args(match::skip(match::name("contiguous"))(shape_transform));
        auto not_layout = match::none_of(match::name("layout"));
        return match::pointwise(
            match::used_once(), not_layout, input_has_shape_transform, output_not_pointwise);
    }

    static bool is_shape_transform(instruction_ref ins)
    {
        return ins->inputs().size() == 1 and
               (contains(shape_transforms(), ins->name()) or ins->name() == "contiguous");
    }

    static bool can_fuse_unary(instruction_ref ins)
    {
        return ins->name() == "@literal" or
               ins->get_operator().attributes().contains("pointwise") or
               contains(ins->name(), "reduce");
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        if(ins->outputs().empty())
            return;
        auto input  = ins->inputs().front();
        auto output = ins->outputs().front();

        auto insert_ops = [&](const auto& ops, instruction_ref z) {
            for(const auto& op : ops)
            {
                z = m.insert_instruction(ins, op, z);
            }
            return z;
        };

        std::vector<operation> xops;
        auto x = input;
        while(is_shape_transform(x))
        {
            xops.push_back(x->get_operator());
            x = x->inputs().front();
        }
        std::reverse(xops.begin(), xops.end());

        std::vector<operation> yops;
        auto y              = output;
        auto last_transform = m.end();
        while(is_shape_transform(y) and y->outputs().size() == 1)
        {
            yops.push_back(y->get_operator());
            last_transform = y;
            y              = y->outputs().front();
        }

        bool move_up   = can_fuse_unary(x);
        bool move_down = can_fuse_unary(y);

        if(move_up and move_down)
        {
            if(x->name() == "@literal")
                move_down = false; // NOLINT(bugprone-branch-clone)
            else if(yops.empty())
                move_up = false;
            else
                move_down = false;
        }
        else if(not move_up and not move_down)
        {
            if(not yops.empty())
                move_up = true;
        }

        if(move_up)
        {
            auto z = m.insert_instruction(ins, ins->get_operator(), x);
            z      = insert_ops(xops, z);
            m.replace_instruction(ins, z);
        }
        else if(move_down and not yops.empty())
        {
            auto z = insert_ops(yops, input);
            m.replace_instruction(last_transform, ins->get_operator(), z);
        }
    }
};

struct find_slice_transpose
{
    auto matcher() const
    {
        auto transpose = match::output(match::name("transpose"));
        return match::any(match::any_of[match::outputs()](match::name("slice")(transpose)));
    }

    static std::vector<int64_t> find_common_perm(const std::vector<instruction_ref>& transposes)
    {
        std::map<std::vector<int64_t>, int64_t> count;
        for(auto t : transposes)
        {
            auto perm = t->get_operator().to_value()["permutation"].to_vector<int64_t>();
            count[perm]++;
        }
        return std::max_element(
                   count.begin(), count.end(), by(std::less<>{}, [](auto&& p) { return p.second; }))
            ->first;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        std::vector<instruction_ref> splits;
        std::copy_if(ins->outputs().begin(),
                     ins->outputs().end(),
                     std::back_inserter(splits),
                     [&](instruction_ref out) {
                         return out->name() == "slice" and out->outputs().size() == 1 and
                                out->outputs().front()->name() == "transpose";
                     });
        if(splits.size() < 2)
            return;
        std::vector<instruction_ref> transposes;
        std::transform(splits.begin(),
                       splits.end(),
                       std::back_inserter(transposes),
                       [](auto split) { return split->outputs().front(); });
        auto perm  = find_common_perm(transposes);
        auto iperm = invert_permutation(perm);
        auto pre   = m.insert_instruction(
            std::next(ins), make_op("transpose", {{"permutation", perm}}), ins);
        for(auto i : range(transposes.size()))
        {
            auto split = splits[i];
            auto t     = transposes[i];
            auto op    = any_cast<op::slice>(split->get_operator());
            std::transform(op.axes.begin(), op.axes.end(), op.axes.begin(), [&](auto axis) {
                return iperm[axis];
            });
            auto new_ins = m.insert_instruction(t, op, pre);
            if(t->get_operator() != pre->get_operator())
            {
                auto curr = t->get_operator().to_value()["permutation"].to_vector<int64_t>();
                new_ins   = m.insert_instruction(
                    t, make_op("transpose", {{"permutation", reorder_dims(iperm, curr)}}), new_ins);
            }
            m.replace_instruction(t, new_ins);
        }
    }
};

struct find_transpose_slice
{
    auto matcher() const
    {
        return match::name("transpose")(match::all_of[match::outputs()](match::name("slice")));
    }

    static std::vector<int64_t> slice_distance(const op::slice& op)
    {
        assert(op.starts.size() == op.ends.size());
        std::vector<int64_t> result(op.starts.size());
        std::transform(
            op.ends.begin(), op.ends.end(), op.starts.begin(), result.begin(), std::minus<>{});
        return result;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins    = r.result;
        auto slices = ins->outputs();
        if(slices.empty())
            return;
        auto slice     = any_cast<op::slice>(slices.front()->get_operator());
        auto sdistance = slice_distance(slice);
        // Check all distances and axes are the same
        if(std::any_of(slices.begin(), slices.end(), [&](auto sins) {
               auto s = any_cast<op::slice>(sins->get_operator());
               return s.axes != slice.axes or slice_distance(s) != sdistance;
           }))
            return;
        // Check distances are divisible by lens of corresponding axes
        auto mod_by_distance = [&](const auto& v, auto f) {
            return std::inner_product(v.begin(),
                                      v.end(),
                                      sdistance.begin(),
                                      0,
                                      std::plus<>{},
                                      [&](auto x, auto d) -> uint64_t {
                                          if(d == 0)
                                              return 1;
                                          return f(x) % d;
                                      });
        };
        if(mod_by_distance(slice.axes, [&](auto x) { return ins->get_shape().lens()[x]; }) != 0 or
           mod_by_distance(slice.starts, id{}) != 0 or mod_by_distance(slice.ends, id{}) != 0)
            return;
        // TODO: Handle multiple axes
        if(sdistance.size() != 1)
            return;
        auto axis = slice.axes.front();
        // Skip if axis would be packed
        if(std::all_of(ins->get_shape().lens().begin(),
                       ins->get_shape().lens().begin() + axis,
                       [](auto x) { return x == 1; }))
            return;
        // Compute axis before transpose to use for unsqueeze
        auto perm    = ins->get_operator().to_value()["permutation"].to_vector<int64_t>();
        auto preaxis = perm[axis];
        // Make unsqueeze
        std::vector<int64_t> steps(sdistance.size());
        std::transform(
            slice.axes.begin(),
            slice.axes.end(),
            sdistance.begin(),
            steps.begin(),
            [&](const auto ax, const auto sdis) { return ins->get_shape().lens().at(ax) / sdis; });
        auto unsqueeze = m.insert_instruction(
            ins, make_op("unsqueeze", {{"axes", {preaxis}}, {"steps", steps}}), ins->inputs());
        // Make transpose
        std::transform(perm.begin(), perm.end(), perm.begin(), [&](auto i) {
            if(i >= preaxis)
                return i + 1;
            return i;
        });
        perm.insert(perm.begin(), preaxis);
        auto transpose =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), unsqueeze);
        // Slice and squeeze
        for(auto s : slices)
        {
            auto op        = any_cast<op::slice>(s->get_operator());
            op.axes        = {0};
            op.starts      = {op.starts.front() / sdistance.front()};
            op.ends        = {op.ends.front() / sdistance.front()};
            auto slice_ins = m.insert_instruction(ins, op, transpose);
            auto squeeze =
                m.insert_instruction(ins, make_op("squeeze", {{"axes", {0}}}), slice_ins);
            m.replace_instruction(s, squeeze);
        }
    }
};

struct find_reshape_dot
{
    auto matcher() const
    {
        auto rsp   = match::name("reshape").bind("rsp");
        auto other = match::skip_broadcasts(match::any().bind("other"));
        return match::name("dot")(match::used_once(), match::either_arg(0, 1)(rsp, other));
    }

    // Gemm axis should not be altered by the reshape
    auto is_valid_reshape(instruction_ref inp, instruction_ref rsp, size_t dot_axis) const
    {
        auto inp_lens = inp->get_shape().lens();
        auto rsp_lens = rsp->get_shape().lens();

        return (inp_lens.size() >= dot_axis and
                rsp_lens[rsp_lens.size() - dot_axis] == inp_lens[inp_lens.size() - dot_axis]);
    }

    // Same batch dims
    auto has_same_batch_dims(instruction_ref in1, instruction_ref in2) const
    {
        auto in1_lens = in1->get_shape().lens();
        auto in2_lens = in2->get_shape().lens();

        return (
            in1_lens.size() == in2_lens.size() and
            std::equal(in1_lens.begin(), in1_lens.end() - 2, in2_lens.begin(), in2_lens.end() - 2));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto dot   = r.result;
        auto rsp   = r.instructions["rsp"];
        auto other = r.instructions["other"];

        auto rsp_lens = rsp->get_shape().lens();
        auto inp      = rsp->inputs().front();
        auto inp_lens = inp->get_shape().lens();

        // Gemm axis should not be altered by the reshape
        bool flipped    = rsp == dot->inputs().back();
        size_t dot_axis = (flipped) ? 2 : 1;

        if(not is_valid_reshape(inp, rsp, dot_axis))
            return;

        instruction_ref new_other;
        if(other->get_operator().name() == "reshape")
        {
            auto other_inp        = other->inputs().front();
            size_t other_dot_axis = (flipped) ? 1 : 2;
            if(not is_valid_reshape(other_inp, other, other_dot_axis) or
               not has_same_batch_dims(inp, other_inp))
                return;

            new_other = other_inp;
        }
        else
        {
            auto other_lens = other->get_shape().lens();
            if(other_lens.size() > 2)
                return;

            std::vector<size_t> new_other_lens{inp_lens.begin(), inp_lens.end() - 2};
            operation new_bc_op;

            auto bc_other      = (flipped) ? dot->inputs().front() : dot->inputs().back();
            auto bc_other_lens = bc_other->get_shape().lens();
            new_other_lens.insert(
                new_other_lens.end(), bc_other_lens.end() - 2, bc_other_lens.end());

            // if the original weight is one dimensional, look at the original broadcast
            // to determine the correct broadcast axis
            if(other_lens.size() == 1)
            {
                auto bc_other_strides = bc_other->get_shape().strides();
                auto it               = std::find_if(bc_other_strides.begin(),
                                       bc_other_strides.end(),
                                       [&](auto i) { return i != 0; });
                auto orig_bc_axis     = std::distance(bc_other_strides.begin(), it);

                auto new_bc_axis = new_other_lens.size() - (bc_other_lens.size() - orig_bc_axis);
                new_bc_op =
                    make_op("broadcast", {{"axis", new_bc_axis}, {"out_lens", new_other_lens}});
            }
            else
            {
                new_bc_op = make_op("multibroadcast", {{"out_lens", new_other_lens}});
            }

            new_other = m.insert_instruction(dot, new_bc_op, other);
        }

        instruction_ref new_dot;
        if(flipped)
        {
            new_dot = m.insert_instruction(dot, make_op("dot"), new_other, inp);
        }
        else
        {
            new_dot = m.insert_instruction(dot, make_op("dot"), inp, new_other);
        }
        m.replace_instruction(
            dot, make_op("reshape", {{"dims", dot->get_shape().lens()}}), new_dot);
    }
};

// Remove transposes and converts between mul/add -> dot so simplify_algebra can perform
// const folding simplifications
struct find_mul_add_shape_op_dot
{
    auto matcher() const
    {
        auto shape_ops             = match::name("transpose", "convert");
        auto const_mul_add         = match::name("mul", "add")(match::either_arg(0, 1)(
            match::is_constant().bind("const"), match::any().bind("input")));
        auto match_shape_op        = shape_ops(match::args(const_mul_add.bind("pw")));
        auto skip_shape_op_outputs = match::skip_output(match::any_of(shape_ops));
        return match_shape_op(skip_shape_op_outputs(match::name("dot")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto shape_ins = r.result;
        auto pw        = r.instructions["pw"];
        auto constant  = r.instructions["const"];
        auto input     = r.instructions["input"];

        auto shape_op  = shape_ins->get_operator();
        auto pw_op     = pw->get_operator();
        auto new_inp   = m.insert_instruction(shape_ins, shape_op, input);
        auto new_const = m.insert_instruction(shape_ins, shape_op, constant);

        m.replace_instruction(shape_ins, pw_op, new_inp, new_const);
    }
};

struct find_flatten
{
    auto matcher() const { return match::name("flatten"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto flatten = r.result;
        m.replace_instruction(flatten,
                              make_op("reshape", {{"dims", flatten->get_shape().lens()}}),
                              flatten->inputs());
    }
};
} // namespace

void simplify_reshapes::apply(module& m) const
{
    if(enable_gather_rewrite)
        match::find_matches(m, find_gather{});
    m.repeat_while_changes(depth, [&] {
        match::find_matches(m,
                            find_nop_reshapes{},
                            find_flatten{},
                            find_reshape_cont{},
                            find_slice_shape_transforms{},
                            find_nested_shape_transforms{},
                            find_concat_slice{},
                            find_concat_transpose{},
                            find_concat_reshape{},
                            find_concat_multibroadcasts{},
                            find_nested_slice{},
                            find_nested_concat{},
                            find_transpose_slice{},
                            find_slice_transpose{},
                            find_unary_shape_transforms{},
                            find_reshape_dot{},
                            find_mul_add_shape_op_dot{},
                            find_op_shape_transform_op{.enable = enable_op_shape_transform_op});
        dead_code_elimination{}.apply(m);
    });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
