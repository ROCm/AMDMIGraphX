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
#include <iterator>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/op/squeeze.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/op/gather.hpp>
#include <migraphx/argument.hpp>
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

#include <map>
#include <limits>
#include <numeric>
#include <optional>
#include <variant>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
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
            auto y = x;
            for(const auto& op : opt_ops)
                y = m.insert_instruction(ins, op, y);
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
        return not desc.has_broadcast();
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
        auto desc1 = shape_transform_descriptor::create(x_ins->get_shape().lens(), ops);
        auto desc  = desc1.rebase(x_ins->inputs().front()->get_shape().lens(), true);
        if(not desc.empty())
            return desc;
        if(not is_reduce(x_ins))
            return desc1;
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
        return shape_transform_descriptor::create(x_ins->get_shape().lens(), ops)
            .rebase(x_ins->inputs().front()->get_shape().lens(), true);
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

struct find_resize
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::name("reshape").bind("data"), match::is_constant().bind("ind")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto ins_rsp = r.instructions["data"];
        auto ins_ind = r.instructions["ind"];

        // resize input shape
        if(ins_rsp->get_shape().lens().size() != 1)
        {
            return;
        }

        // resize output shape
        const auto& in_shape  = ins_rsp->inputs().front()->get_shape();
        const auto& out_shape = ins->get_shape();
        // check if output shape is multiple of input shape
        const auto& in_lens  = in_shape.lens();
        const auto& out_lens = out_shape.lens();
        if(in_lens.size() != out_lens.size())
        {
            return;
        }

        // output shape must be multiple of input shape
        std::vector<bool> is_multi(in_lens.size());
        std::transform(
            in_lens.begin(), in_lens.end(), out_lens.begin(), is_multi.begin(), [](auto x, auto y) {
                return (y % x == 0);
            });
        if(not std::all_of(is_multi.begin(), is_multi.end(), [](auto b) { return b; }))
        {
            return;
        }

        // output must be multiple of inputs
        std::vector<std::size_t> scales(in_lens.size());
        std::transform(
            in_lens.begin(), in_lens.end(), out_lens.begin(), scales.begin(), [](auto x, auto y) {
                return y / x;
            });

        // if ind is not constant, cannot optimize
        std::vector<int> vec_ind;
        auto arg_ind = ins_ind->eval();
        if(arg_ind.empty())
        {
            return;
        }
        arg_ind.visit([&](auto v) { vec_ind.assign(v.begin(), v.end()); });
        if(not all_of(range(out_shape.elements()), [&](auto i) {
               auto out_idx = out_shape.multi(i);
               auto in_idx  = out_idx;
               std::transform(out_idx.begin(),
                              out_idx.end(),
                              scales.begin(),
                              in_idx.begin(),
                              [&](auto io, auto scale) { return io - (io % scale); });
               return vec_ind[i] == vec_ind[out_shape.index(in_idx)];
           }))
        {
            return;
        }

        // wrap up shapes for multibroadcast
        std::vector<std::pair<std::size_t, std::size_t>> dim_scales;
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       out_lens.begin(),
                       std::back_inserter(dim_scales),
                       [](auto x, auto y) { return std::make_pair(x, y / x); });

        std::vector<int64_t> in_dims;
        std::vector<int64_t> out_dims;
        for(auto& isp : dim_scales)
        {
            in_dims.push_back(isp.first);
            out_dims.push_back(isp.first * isp.second);
            if(isp.first == 1 or isp.second == 1)
            {
                continue;
            }

            out_dims.back() = isp.first;
            in_dims.push_back(1);
            out_dims.push_back(isp.second);
        }

        auto in_rsp   = ins_rsp->inputs().front();
        auto rsp_data = m.insert_instruction(
            ins_rsp, migraphx::make_op("reshape", {{"dims", in_dims}}), in_rsp);
        auto mb_rsp = m.insert_instruction(
            ins_rsp, migraphx::make_op("multibroadcast", {{"out_lens", out_dims}}), rsp_data);
        std::vector<int64_t> rsp_dims(out_lens.begin(), out_lens.end());
        m.replace_instruction(ins, migraphx::make_op("reshape", {{"dims", rsp_dims}}), mb_rsp);
    }
};

struct find_gather
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::any(), match::is_constant().bind("indices")));
    }

private:
    using index_list = std::vector<std::size_t>;

    static std::vector<int64_t> to_int64(const std::vector<std::size_t>& values)
    {
        std::vector<int64_t> result;
        result.reserve(values.size());
        std::transform(values.begin(),
                       values.end(),
                       std::back_inserter(result),
                       [](std::size_t v) { return static_cast<int64_t>(v); });
        return result;
    }

    template <class Container>
    static std::vector<int64_t> to_int64_container(const Container& values)
    {
        std::vector<int64_t> result;
        result.reserve(values.size());
        std::transform(values.begin(),
                       values.end(),
                       std::back_inserter(result),
                       [](auto v) { return static_cast<int64_t>(v); });
        return result;
    }

    template <class Iter>
    static std::size_t product(Iter first, Iter last)
    {
        return std::accumulate(first, last, std::size_t{1}, std::multiplies<>{});
    }

    static std::size_t product(const std::vector<std::size_t>& lens)
    {
        return product(lens.begin(), lens.end());
    }

    static bool is_constant_range(std::vector<std::size_t>::const_iterator first,
                                  std::vector<std::size_t>::const_iterator last)
    {
        if(first == last)
            return false;
        return std::adjacent_find(first, last, std::not_equal_to<>{}) == last;
    }

    static bool is_contiguous_range(std::vector<std::size_t>::const_iterator first,
                                    std::vector<std::size_t>::const_iterator last)
    {
        if(std::distance(first, last) <= 1)
            return false;
        const auto start = *first;
        std::size_t expected = 0;
        return std::all_of(first, last, [&](std::size_t value) mutable {
            const auto want = start + expected;
            ++expected;
            return value == want;
        });
    }

    static std::optional<std::size_t>
    arithmetic_stride(std::vector<std::size_t>::const_iterator first,
                      std::vector<std::size_t>::const_iterator last)
    {
        const auto size = std::distance(first, last);
        if(size <= 1)
            return std::nullopt;
        auto base   = static_cast<std::int64_t>(*first);
        auto second = static_cast<std::int64_t>(*std::next(first));
        auto stride = second - base;
        if(stride <= 1)
            return std::nullopt;
        std::int64_t value = second;
        for(auto it = std::next(first, 2); it != last; ++it)
        {
            value += stride;
            if(static_cast<std::int64_t>(*it) != value)
                return std::nullopt;
        }
        return static_cast<std::size_t>(stride);
    }

    struct normalized_gather
    {
        module* pm                         = nullptr;
        instruction_ref anchor             = instruction_ref{};
        instruction_ref data_aligned       = instruction_ref{};
        instruction_ref data_axis_view     = instruction_ref{};
        std::vector<std::size_t> pre_lens  = {};
        std::vector<std::size_t> post_lens = {};
        std::vector<std::size_t> rest_lens = {};
        std::vector<std::size_t> indices_lens{};
        std::vector<std::size_t> indices_flat{};
        std::vector<std::size_t> idims{};
        std::vector<std::int64_t> indices_values{};
        std::size_t axis_len     = 0;
        std::size_t rest_product = 1;
        std::size_t axis_index   = 0;
        instruction_ref data_ins{};

        module& mod() const { return *pm; }

        std::vector<std::size_t> reshape_dims() const
        {
            auto dims = indices_lens;
            dims.insert(dims.end(), rest_lens.begin(), rest_lens.end());
            return dims;
        }

        std::vector<int64_t> final_permutation() const
        {
            const auto pre  = pre_lens.size();
            const auto post = post_lens.size();
            if(pre == 0)
                return {};
            const auto indices = indices_lens.size();
            std::vector<int64_t> perm(indices + pre + post);
            std::size_t pos = 0;
            for(std::size_t i = 0; i < pre; ++i)
                perm[pos++] = static_cast<int64_t>(indices + i);
            for(std::size_t i = 0; i < indices; ++i)
                perm[pos++] = static_cast<int64_t>(i);
            for(std::size_t i = 0; i < post; ++i)
                perm[pos++] = static_cast<int64_t>(indices + pre + i);
            bool identity = true;
            for(std::size_t i = 0; i < perm.size(); ++i)
            {
                if(perm[i] != static_cast<int64_t>(i))
                {
                    identity = false;
                    break;
                }
            }
            if(identity)
                return {};
            return perm;
        }
    };

    static std::optional<normalized_gather> normalize(module& m,
                                                      instruction_ref gather_ins,
                                                      instruction_ref data_ins,
                                                      instruction_ref indices_ins)
    {
        normalized_gather ctx;
        ctx.pm          = &m;
        ctx.anchor      = gather_ins;
        ctx.data_ins    = data_ins;
        ctx.indices_lens = indices_ins->get_shape().lens();
        ctx.idims        = ctx.indices_lens;

        auto gather_op = any_cast<op::gather>(gather_ins->get_operator());
        const auto& dlens = data_ins->get_shape().lens();
        if(dlens.empty())
            return std::nullopt;

        ctx.axis_index = static_cast<std::size_t>(
            tune_axis(static_cast<int>(dlens.size()), gather_op.axis, gather_op.name()));
        ctx.axis_len = dlens.at(ctx.axis_index);
        if(ctx.axis_len == 0)
            return std::nullopt;

        auto literal = indices_ins->eval();
        if(literal.empty())
            return std::nullopt;
        ctx.indices_values.resize(literal.get_shape().elements());
        literal.visit([&](auto values) {
            std::transform(values.begin(),
                           values.end(),
                           ctx.indices_values.begin(),
                           [](auto v) { return static_cast<std::int64_t>(v); });
        });
        if(ctx.indices_values.empty())
            return std::nullopt;

        if(indices_ins->get_shape().elements() != ctx.indices_values.size())
            return std::nullopt;

        ctx.indices_flat.reserve(ctx.indices_values.size());
        const auto axis_len_i = static_cast<std::int64_t>(ctx.axis_len);
        for(auto value : ctx.indices_values)
        {
            auto adjusted = value;
            if(adjusted < 0)
                adjusted += axis_len_i;
            if(adjusted < 0 or adjusted >= axis_len_i)
                return std::nullopt;
            ctx.indices_flat.push_back(static_cast<std::size_t>(adjusted));
        }

        const auto& dl = dlens;
        ctx.pre_lens  = std::vector<std::size_t>(dl.begin(), dl.begin() + ctx.axis_index);
        ctx.post_lens = std::vector<std::size_t>(dl.begin() + ctx.axis_index + 1, dl.end());
        ctx.rest_lens = ctx.pre_lens;
        ctx.rest_lens.insert(ctx.rest_lens.end(), ctx.post_lens.begin(), ctx.post_lens.end());
        ctx.rest_product = ctx.rest_lens.empty() ? 1 : product(ctx.rest_lens);
        if(ctx.rest_product == 0)
            return std::nullopt;

        instruction_ref aligned = data_ins;
        if(ctx.axis_index != 0)
        {
            std::vector<int64_t> perm;
            perm.reserve(dlens.size());
            perm.push_back(static_cast<int64_t>(ctx.axis_index));
            for(std::size_t i = 0; i < dlens.size(); ++i)
            {
                if(i == ctx.axis_index)
                    continue;
                perm.push_back(static_cast<int64_t>(i));
            }
            aligned = m.insert_instruction(gather_ins,
                                           make_op("transpose", {{"permutation", perm}}),
                                           aligned);
        }
        ctx.data_aligned = aligned;
        const std::vector<int64_t> reshape_axis{static_cast<int64_t>(ctx.axis_len),
                                                static_cast<int64_t>(ctx.rest_product)};
        ctx.data_axis_view = m.insert_instruction(
            gather_ins, make_op("reshape", {{"dims", reshape_axis}}), aligned);

        return ctx;
    }

    struct tile_partition
    {
        std::size_t tile_size            = 1;
        std::vector<std::size_t> tile_shape;
        std::vector<std::size_t> outer_shape;
    };

    static std::vector<tile_partition>
    make_partitions(const std::vector<std::size_t>& indices_lens)
    {
        if(indices_lens.empty())
            return {};
        const auto rank = indices_lens.size();
        const auto maxk = std::min<std::size_t>(rank, 3);
        std::vector<tile_partition> partitions;
        partitions.reserve(maxk);
        for(std::size_t k = maxk; k >= 1; --k)
        {
            const auto start = rank - k;
            tile_partition part;
            part.tile_shape = std::vector<std::size_t>(indices_lens.begin() + start, indices_lens.end());
            part.outer_shape = std::vector<std::size_t>(indices_lens.begin(), indices_lens.begin() + start);
            part.tile_size   = product(part.tile_shape.begin(), part.tile_shape.end());
            partitions.push_back(std::move(part));
            if(k == 1)
                break;
        }
        return partitions;
    }

    struct tile_view
    {
        const index_list* values = nullptr;
        std::size_t offset       = 0;
        std::size_t size         = 0;

        auto begin() const { return values->begin() + static_cast<std::ptrdiff_t>(offset); }
        auto end() const { return begin() + static_cast<std::ptrdiff_t>(size); }
        std::size_t front() const { return *(begin()); }
    };

    struct constant_pattern
    {
        std::size_t value = 0;
    };

    struct contiguous_pattern
    {
        std::size_t start = 0;
    };

    struct arithmetic_pattern
    {
        std::size_t start  = 0;
        std::size_t stride = 0;
        std::size_t lane   = 0;
        std::size_t base   = 0;
    };

    struct rtr_pattern
    {
        std::vector<std::size_t> factors;
        std::vector<std::size_t> permutation;
        std::size_t window_start = 0;
    };

    enum class tile_kind
    {
        constant,
        contiguous,
        arithmetic,
        rtr
    };

    struct tile_plan
    {
        tile_view view{};
        tile_kind kind = tile_kind::constant;
        std::variant<constant_pattern, contiguous_pattern, arithmetic_pattern, rtr_pattern> pattern =
            constant_pattern{};
    };

    static std::optional<std::vector<std::size_t>>
    factorize(std::size_t value, std::size_t max_factors = 8)
    {
        if(value <= 1)
            return std::nullopt;
        std::vector<std::size_t> factors;
        std::size_t remaining = value;
        for(std::size_t p = 2; p * p <= remaining; ++p)
        {
            while(remaining % p == 0)
            {
                factors.push_back(p);
                remaining /= p;
                if(factors.size() > max_factors)
                    return std::nullopt;
            }
        }
        if(remaining > 1)
            factors.push_back(remaining);
        if(factors.size() > max_factors)
            return std::nullopt;
        return factors;
    }

    static std::vector<std::vector<std::size_t>>
    enumerate_factorizations(std::size_t value, std::size_t max_results = 256)
    {
        std::vector<std::vector<std::size_t>> results;
        if(value <= 1)
        {
            results.push_back({value});
            return results;
        }

        std::vector<std::size_t> current;
        const auto dfs = [&](auto&& self, std::size_t remaining, std::size_t min_factor) -> void {
            for(std::size_t f = min_factor; f * f <= remaining; ++f)
            {
                if(remaining % f != 0)
                    continue;
                current.push_back(f);
                self(self, remaining / f, f);
                current.pop_back();
                if(results.size() >= max_results)
                    return;
            }
            if(results.size() >= max_results)
                return;
            current.push_back(remaining);
            results.push_back(current);
            current.pop_back();
        };

        dfs(dfs, value, 2);
        return results;
    }

    static std::vector<std::size_t>
    combine_factors(const std::vector<std::size_t>& primes, std::size_t max_parts = 6)
    {
        if(primes.empty())
            return {};
        std::vector<std::size_t> parts{primes.front()};
        for(std::size_t i = 1; i < primes.size(); ++i)
        {
            if(parts.size() < max_parts)
                parts.push_back(primes[i]);
            else
                parts.back() *= primes[i];
        }
        return parts;
    }

    static std::vector<std::size_t> make_axes(std::size_t n)
    {
        std::vector<std::size_t> axes(n);
        std::iota(axes.begin(), axes.end(), std::size_t{0});
        return axes;
    }

    static std::vector<std::size_t> compute_permutation_order(const std::vector<std::size_t>& dims,
                                                              const std::vector<std::size_t>& permutation)
    {
        std::vector<std::size_t> reordered;
        reordered.reserve(dims.size());
        for(auto axis : permutation)
            reordered.push_back(dims.at(axis));

        std::vector<std::size_t> coord(permutation.size(), 0);
        const auto total = product(dims);
        std::vector<std::size_t> order;
        order.reserve(total);
        for(std::size_t i = 0; i < total; ++i)
        {
            std::vector<std::size_t> orig(permutation.size(), 0);
            for(std::size_t j = 0; j < permutation.size(); ++j)
                orig[permutation[j]] = coord[j];

            std::size_t idx = 0;
            for(std::size_t j = 0; j < dims.size(); ++j)
                idx = idx * dims[j] + orig[j];
            order.push_back(idx);

            std::size_t pos = coord.size();
            while(pos > 0)
            {
                --pos;
                coord[pos]++;
                if(coord[pos] < reordered[pos])
                    break;
                coord[pos] = 0;
            }
        }
        return order;
    }

    static std::optional<rtr_pattern> detect_rtr(const tile_view& view, std::size_t axis_len)
    {
        if(axis_len <= 1)
            return std::nullopt;
        if(view.size == 0)
            return std::nullopt;

        const auto primes = factorize(axis_len);
        if(not primes)
            return std::nullopt;

        std::vector<std::vector<std::size_t>> candidate_dims;
        const auto factorizations = enumerate_factorizations(axis_len);
        candidate_dims.reserve(factorizations.size());
        std::transform(factorizations.begin(),
                       factorizations.end(),
                       std::back_inserter(candidate_dims),
                       [](const auto& factors) {
                           return combine_factors(factors);
                       });

        for(auto& dims : candidate_dims)
        {
            if(dims.empty() or dims.size() > 6)
                continue;
            auto axes = make_axes(dims.size());
            do
            {
                const auto order = compute_permutation_order(dims, axes);
                if(order.size() < view.size)
                    continue;
                auto begin = std::search(order.begin(), order.end(), view.begin(), view.end());
                if(begin == order.end())
                    continue;
                const auto offset = static_cast<std::size_t>(std::distance(order.begin(), begin));
                rtr_pattern pattern;
                pattern.factors      = dims;
                pattern.permutation  = axes;
                pattern.window_start = offset;
                return pattern;
            } while(std::next_permutation(axes.begin(), axes.end()));
        }
        return std::nullopt;
    }

    static std::optional<tile_plan> analyze_tile(const tile_view& view,
                                                 std::size_t axis_len)
    {
        if(view.size == 0)
            return std::nullopt;

        auto first = view.begin();
        auto last  = view.end();

        if(is_constant_range(first, last))
        {
            tile_plan plan;
            plan.view    = view;
            plan.kind    = tile_kind::constant;
            plan.pattern = constant_pattern{view.front()};
            return plan;
        }

        if(is_contiguous_range(first, last))
        {
            tile_plan plan;
            plan.view    = view;
            plan.kind    = tile_kind::contiguous;
            plan.pattern = contiguous_pattern{view.front()};
            return plan;
        }

        if(auto stride = arithmetic_stride(first, last))
        {
            if(axis_len % *stride == 0)
            {
                const auto start = view.front();
                arithmetic_pattern pattern;
                pattern.start  = start;
                pattern.stride = *stride;
                pattern.lane   = start % pattern.stride;
                pattern.base   = start / pattern.stride;
                tile_plan plan;
                plan.view    = view;
                plan.kind    = tile_kind::arithmetic;
                plan.pattern = pattern;
                return plan;
            }
        }

        if(auto pattern = detect_rtr(view, axis_len))
        {
            tile_plan plan;
            plan.view    = view;
            plan.kind    = tile_kind::rtr;
            plan.pattern = *pattern;
            return plan;
        }

        return std::nullopt;
    }

    static std::optional<std::vector<tile_plan>>
    choose_tiles(const normalized_gather& ctx, const tile_partition& partition)
    {
        if(partition.tile_size == 0)
            return std::nullopt;
        if(ctx.indices_flat.size() % partition.tile_size != 0)
            return std::nullopt;

        std::vector<tile_plan> plans;
        plans.reserve(ctx.indices_flat.size() / partition.tile_size);
        for(std::size_t offset = 0; offset < ctx.indices_flat.size(); offset += partition.tile_size)
        {
            tile_view view{&ctx.indices_flat, offset, partition.tile_size};
            auto plan = analyze_tile(view, ctx.axis_len);
            if(not plan)
                return std::nullopt;
            plans.push_back(*plan);
        }
        return plans;
    }

    static std::optional<std::pair<tile_partition, std::vector<tile_plan>>>
    build_plan(const normalized_gather& ctx)
    {
        const auto partitions = make_partitions(ctx.indices_lens);
        for(const auto& partition : partitions)
        {
            auto plans = choose_tiles(ctx, partition);
            if(plans)
                return std::make_pair(partition, *plans);
        }
        std::cerr << "gather: no tiling plan for indices lens";
        for(auto len : ctx.indices_lens)
            std::cerr << " " << len;
        std::cerr << "\n";
        return std::nullopt;
    }

    struct tile_rewriter
    {
        normalized_gather& ctx;
        tile_partition partition;
        std::map<std::size_t, instruction_ref> stride_cache{};
        std::map<std::pair<std::size_t, std::size_t>, instruction_ref> lane_cache{};

        instruction_ref materialize(const std::vector<tile_plan>& plans)
        {
            if(plans.empty())
                return instruction_ref{};
            std::vector<instruction_ref> parts;
            parts.reserve(plans.size());
            std::transform(plans.begin(),
                           plans.end(),
                           std::back_inserter(parts),
                           [&](const tile_plan& plan) { return rewrite_tile(plan); });
            if(parts.empty())
                return instruction_ref{};
            if(parts.size() == 1)
                return parts.front();
            return ctx.mod().insert_instruction(
                ctx.anchor, make_op("concat", {{"axis", int64_t{0}}}), parts);
        }

        instruction_ref rewrite_tile(const tile_plan& plan)
        {
            switch(plan.kind)
            {
            case tile_kind::constant:
                return rewrite_constant(std::get<constant_pattern>(plan.pattern), plan.view.size);
            case tile_kind::contiguous:
                return rewrite_contiguous(std::get<contiguous_pattern>(plan.pattern), plan.view.size);
            case tile_kind::arithmetic:
                return rewrite_arithmetic(std::get<arithmetic_pattern>(plan.pattern), plan.view);
            case tile_kind::rtr:
                return rewrite_rtr(std::get<rtr_pattern>(plan.pattern), plan.view);
            }
            return instruction_ref{};
        }

        instruction_ref rewrite_constant(const constant_pattern& pattern, std::size_t tile_size)
        {
            const auto axes   = std::vector<int64_t>{0};
            const auto starts = std::vector<int64_t>{static_cast<int64_t>(pattern.value)};
            const auto ends   = std::vector<int64_t>{static_cast<int64_t>(pattern.value + 1)};
            auto slice = ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                ctx.data_axis_view);
            if(tile_size == 1)
                return slice;
            const std::vector<int64_t> out_lens{static_cast<int64_t>(tile_size),
                                                static_cast<int64_t>(ctx.rest_product)};
            return ctx.mod().insert_instruction(
                ctx.anchor, make_op("multibroadcast", {{"out_lens", out_lens}}), slice);
        }

        instruction_ref rewrite_contiguous(const contiguous_pattern& pattern, std::size_t size)
        {
            const auto axes   = std::vector<int64_t>{0};
            const auto starts = std::vector<int64_t>{static_cast<int64_t>(pattern.start)};
            const auto ends   = std::vector<int64_t>{static_cast<int64_t>(pattern.start + size)};
            return ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                ctx.data_axis_view);
        }

        instruction_ref ensure_stride_view(std::size_t stride)
        {
            auto it = stride_cache.find(stride);
            if(it != stride_cache.end())
                return it->second;
            std::vector<int64_t> dims{static_cast<int64_t>(ctx.axis_len / stride),
                                      static_cast<int64_t>(stride),
                                      static_cast<int64_t>(ctx.rest_product)};
            auto view = ctx.mod().insert_instruction(
                ctx.anchor, make_op("reshape", {{"dims", dims}}), ctx.data_axis_view);
            stride_cache.emplace(stride, view);
            return view;
        }

        instruction_ref ensure_lane_view(std::size_t stride, std::size_t lane)
        {
            const auto key = std::make_pair(stride, lane);
            auto it        = lane_cache.find(key);
            if(it != lane_cache.end())
                return it->second;
            auto stride_view = ensure_stride_view(stride);
            const std::vector<int64_t> axes{1};
            const std::vector<int64_t> starts{static_cast<int64_t>(lane)};
            const std::vector<int64_t> ends{static_cast<int64_t>(lane + 1)};
            auto slice = ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                stride_view);
            const std::vector<int64_t> squeeze_axes{1};
            auto squeezed = ctx.mod().insert_instruction(
                ctx.anchor, make_op("squeeze", {{"axes", squeeze_axes}}), slice);
            lane_cache.emplace(key, squeezed);
            return squeezed;
        }

        instruction_ref rewrite_arithmetic(const arithmetic_pattern& pattern, const tile_view& view)
        {
            auto lane_view = ensure_lane_view(pattern.stride, pattern.lane);
            const auto axes   = std::vector<int64_t>{0};
            const auto starts = std::vector<int64_t>{static_cast<int64_t>(pattern.base)};
            const auto ends =
                std::vector<int64_t>{static_cast<int64_t>(pattern.base + view.size)};
            return ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                lane_view);
        }

        instruction_ref rewrite_rtr(const rtr_pattern& pattern, const tile_view& view)
        {
            std::vector<int64_t> reshape_dims;
            reshape_dims.reserve(pattern.factors.size() + 1);
            for(auto dim : pattern.factors)
                reshape_dims.push_back(static_cast<int64_t>(dim));
            reshape_dims.push_back(static_cast<int64_t>(ctx.rest_product));
            auto reshaped = ctx.mod().insert_instruction(
                ctx.anchor, make_op("reshape", {{"dims", reshape_dims}}), ctx.data_axis_view);

            std::vector<int64_t> perm(pattern.permutation.begin(), pattern.permutation.end());
            perm.push_back(static_cast<int64_t>(pattern.permutation.size()));
            auto transposed = ctx.mod().insert_instruction(
                ctx.anchor, make_op("transpose", {{"permutation", perm}}), reshaped);

            const std::vector<int64_t> axes{0};
            const std::vector<int64_t> starts{static_cast<int64_t>(pattern.window_start)};
            const std::vector<int64_t> ends{static_cast<int64_t>(pattern.window_start + view.size)};
            auto sliced = ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                transposed);

            std::vector<int64_t> reshape_back{static_cast<int64_t>(view.size),
                                               static_cast<int64_t>(ctx.rest_product)};
            return ctx.mod().insert_instruction(
                ctx.anchor, make_op("reshape", {{"dims", reshape_back}}), sliced);
        }
    };

    static instruction_ref finalize_output(normalized_gather& ctx,
                                           instruction_ref flat_view,
                                           const std::vector<std::size_t>& reshape_dims)
    {
        if(flat_view == instruction_ref{})
            return instruction_ref{};
        auto reshaped = ctx.mod().insert_instruction(
            ctx.anchor, make_op("reshape", {{"dims", to_int64_container(reshape_dims)}}), flat_view);
        auto perm = ctx.final_permutation();
        instruction_ref final_out = reshaped;
        if(not perm.empty())
        {
            final_out = ctx.mod().insert_instruction(
                ctx.anchor, make_op("transpose", {{"permutation", perm}}), reshaped);
        }

        if(final_out->get_shape() != ctx.anchor->get_shape())
        {
            final_out = ctx.mod().insert_instruction(
                ctx.anchor,
                make_op("reshape",
                        {{"dims", to_int64_container(ctx.anchor->get_shape().lens())}}),
                final_out);
        }
        return final_out;
    }

public:
    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto data_ins    = ins->inputs().front();
        auto indices_ins = r.instructions["indices"];

        auto ctx_opt = normalize(m, ins, data_ins, indices_ins);
        if(not ctx_opt)
            return;
        auto& ctx = *ctx_opt;

        auto plan = build_plan(ctx);
        if(not plan)
            return;

        tile_rewriter rewriter{ctx, plan->first};
        auto flat = rewriter.materialize(plan->second);
        auto result = finalize_output(ctx, flat, ctx.reshape_dims());
        if(result == instruction_ref{})
            return;
        ctx.mod().replace_instruction(ins, result);
    }
};


struct find_where_op
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::name("reshape")(match::arg(0)(match::name("concat").bind("data"))),
                        match::is_constant().bind("ind")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto concat  = r.instructions["data"];
        auto ins_ind = r.instructions["ind"];
        std::vector<bool> vec_ind;
        auto arg_ind = ins_ind->eval();
        arg_ind.visit([&](auto v) { vec_ind.assign(v.begin(), v.end()); });
        // ind has to be the same value
        auto val = vec_ind.front();
        if(not std::all_of(vec_ind.begin(), vec_ind.end(), [&](auto v) { return (v == val); }))
        {
            return;
        }

        // concat axis must be 0
        auto op = any_cast<op::concat>(concat->get_operator());
        if(op.axis != 0)
        {
            return;
        }

        // check concat inputs, it has to be 2 and have the same shape
        const auto& inputs = concat->inputs();
        if(inputs.size() != 2)
        {
            return;
        }
        if(inputs.at(0)->get_shape() != inputs.at(1)->get_shape())
        {
            return;
        }
        if(inputs.at(0)->get_shape().lens() != ins_ind->get_shape().lens())
        {
            return;
        }

        if(val)
        {
            m.replace_instruction(ins, inputs.at(0));
        }
        else
        {
            m.replace_instruction(ins, inputs.at(1));
        }
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
    m.repeat_while_changes(depth, [&] {
        match::find_matches(m,
                            find_where_op{},
                            // find_resize{},
                            find_gather{},
                            find_nop_reshapes{},
                            find_flatten{},
                            find_reshape_cont{},
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
