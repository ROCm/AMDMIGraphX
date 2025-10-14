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

#include <array>
#include <map>
#include <cassert>
#include <numeric>
#include <set>
#include <limits>
#include <variant>
#include <memory>

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

// ============================================================================
// Gather Optimization - Utility Functions
// ============================================================================

namespace {

/// Convert vector of sizes to vector of int64
inline std::vector<int64_t> to_int64_vec(const std::vector<std::size_t>& lens)
{
    std::vector<int64_t> result;
    result.reserve(lens.size());
    std::transform(lens.begin(), lens.end(), std::back_inserter(result), [](auto len) {
        return static_cast<int64_t>(len);
    });
    return result;
}

/// Compute product of elements
inline std::size_t product_of(const std::vector<std::size_t>& lens)
{
    return std::accumulate(
        lens.begin(), lens.end(), std::size_t{1}, [](auto acc, auto v) { return acc * v; });
}

/// Factorize a positive integer into prime factors
inline std::vector<std::size_t> factorize_number(std::size_t value)
{
    std::vector<std::size_t> factors;
    auto n = value;
    for(std::size_t p = 2; p * p <= n; ++p)
    {
        while(n % p == 0)
        {
            factors.push_back(p);
            n /= p;
        }
    }
    if(n > 1)
        factors.push_back(n);
    return factors;
}

/// Check if permutation is identity
inline bool is_identity_perm(const std::vector<int64_t>& perm)
{
    return std::all_of(perm.begin(), perm.end(), [i = std::size_t{0}](auto p) mutable {
        return static_cast<std::size_t>(p) == i++;
    });
}

/// Build permutation that moves axis to front
inline std::vector<int64_t> move_axis_to_front_perm(std::size_t axis, std::size_t ndims)
{
    std::vector<int64_t> perm;
    perm.reserve(ndims);
    perm.push_back(static_cast<int64_t>(axis));
    for(std::size_t i = 0; i < ndims; ++i)
    {
        if(i != axis)
            perm.push_back(static_cast<int64_t>(i));
    }
    return perm;
}

/// Build permutation to restore axis position
inline std::vector<int64_t>
restore_axis_position_perm(std::size_t pre_count, std::size_t block_count, std::size_t post_count)
{
    std::vector<int64_t> perm;
    perm.reserve(pre_count + block_count + post_count);

    for(std::size_t i = 0; i < pre_count; ++i)
        perm.push_back(static_cast<int64_t>(block_count + i));
    for(std::size_t i = 0; i < block_count; ++i)
        perm.push_back(static_cast<int64_t>(i));
    for(std::size_t i = 0; i < post_count; ++i)
        perm.push_back(static_cast<int64_t>(block_count + pre_count + i));

    return perm;
}

/// Generate all factorizations using DFS
inline std::vector<std::vector<std::size_t>> enumerate_all_factorizations(std::size_t value,
                                                                          std::size_t max_results)
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
            if(remaining % f != 0 or results.size() >= max_results)
                continue;
            current.push_back(f);
            self(self, remaining / f, f);
            current.pop_back();
            if(results.size() >= max_results)
                return;
        }
        if(not current.empty() and results.size() < max_results)
        {
            current.push_back(remaining);
            results.push_back(current);
            current.pop_back();
        }
    };

    dfs(dfs, value, 2);
    if(results.size() < max_results)
        results.push_back({value});
    return results;
}

/// Build and add unique factorization candidates
inline void add_unique_factorization(std::vector<std::vector<std::size_t>>& candidates,
                                     std::vector<std::size_t> factors,
                                     std::size_t expected_product,
                                     std::size_t max_size)
{
    if(factors.empty() or product_of(factors) != expected_product)
        return;

    factors.erase(std::remove(factors.begin(), factors.end(), std::size_t{1}), factors.end());
    if(factors.empty())
        return;

    if(factors.size() > 8 or candidates.size() >= max_size)
        return;

    if(std::find(candidates.begin(), candidates.end(), factors) == candidates.end())
        candidates.push_back(std::move(factors));
}

// ============================================================================
// Gather Optimization - Helper Classes
// ============================================================================

/// Helper class to build instruction sequences with common patterns
class gather_instruction_builder
{
    module& m;
    instruction_ref insert_before;

    public:
    gather_instruction_builder(module& mod, instruction_ref ins) : m(mod), insert_before(ins) {}

    instruction_ref transpose(instruction_ref input, const std::vector<int64_t>& perm)
    {
        if(is_identity_perm(perm))
            return input;
        return m.insert_instruction(
            insert_before, make_op("transpose", {{"permutation", perm}}), input);
    }

    instruction_ref reshape(instruction_ref input, const std::vector<int64_t>& dims)
    {
        return m.insert_instruction(insert_before, make_op("reshape", {{"dims", dims}}), input);
    }

    instruction_ref unsqueeze(instruction_ref input,
                              const std::vector<int64_t>& axes,
                              const std::vector<int64_t>& steps = {})
    {
        return m.insert_instruction(
            insert_before, make_op("unsqueeze", {{"axes", axes}, {"steps", steps}}), input);
    }

    instruction_ref slice(instruction_ref input,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& starts,
                          const std::vector<int64_t>& ends)
    {
        return m.insert_instruction(
            insert_before,
            make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
            input);
    }

    instruction_ref
    step(instruction_ref input, const std::vector<int64_t>& axes, const std::vector<int64_t>& steps)
    {
        return m.insert_instruction(
            insert_before, make_op("step", {{"axes", axes}, {"steps", steps}}), input);
    }

    instruction_ref slice_with_step(instruction_ref input,
                                    const std::vector<int64_t>& axes,
                                    const std::vector<int64_t>& starts,
                                    const std::vector<int64_t>& ends,
                                    const std::vector<int64_t>& steps)
    {
        auto sliced = slice(input, axes, starts, ends);
        return step(sliced, axes, steps);
    }

    instruction_ref multibroadcast(instruction_ref input, const std::vector<int64_t>& out_lens)
    {
        return m.insert_instruction(
            insert_before, make_op("multibroadcast", {{"out_lens", out_lens}}), input);
    }

    instruction_ref concat(const std::vector<instruction_ref>& inputs, int64_t axis)
    {
        return m.insert_instruction(insert_before, make_op("concat", {{"axis", axis}}), inputs);
    }

    instruction_ref move_axis_to_front(instruction_ref input, std::size_t axis)
    {
        const auto& lens = input->get_shape().lens();
        if(axis == 0)
            return input;
        return transpose(input, move_axis_to_front_perm(axis, lens.size()));
    }

    instruction_ref restore_axis_position(instruction_ref input,
                                          std::size_t pre_count,
                                          std::size_t block_count,
                                          std::size_t post_count)
    {
        auto perm = restore_axis_position_perm(pre_count, block_count, post_count);
        return transpose(input, perm);
    }

    instruction_ref match_shape(instruction_ref input, const std::vector<std::size_t>& target_lens)
    {
        const auto& curr_lens = input->get_shape().lens();
        if(curr_lens == target_lens)
            return input;

        if(input->get_shape().elements() == product_of(target_lens))
            return reshape(input, to_int64_vec(target_lens));

        return multibroadcast(input, to_int64_vec(target_lens));
    }
};

// ============================================================================
// Gather Optimization - Context and Pattern Classes
// ============================================================================

/// Encapsulates all analyzed gather properties
struct gather_context
{
    private:
    instruction_ref data_ins_;
    std::size_t axis_index_;
    argument indices_arg_;
    mutable std::shared_ptr<std::vector<std::vector<std::size_t>>> factor_candidates_;

    public:
    gather_context(instruction_ref data_input, std::size_t axis_idx, argument indices)
        : data_ins_(data_input), axis_index_(axis_idx), indices_arg_(std::move(indices))
    {
    }

    // Accessors for stored values
    instruction_ref data_ins() const { return data_ins_; }
    std::size_t axis_index() const { return axis_index_; }
    const argument& indices_arg() const { return indices_arg_; }

    // Computed properties
    std::vector<int64_t> indices_values() const
    {
        std::vector<std::int64_t> values;
        indices_arg_.visit([&](auto v) {
            values.resize(v.size());
            std::transform(v.begin(), v.end(), values.begin(), [](auto x) {
                return static_cast<std::int64_t>(x);
            });
        });
        return values;
    }

    std::size_t axis_len() const { return data_ins_->get_shape().lens().at(axis_index_); }

    std::vector<std::size_t> pre_lens() const
    {
        const auto& dlens = data_ins_->get_shape().lens();
        return std::vector<std::size_t>(dlens.begin(), dlens.begin() + axis_index_);
    }

    std::vector<std::size_t> post_lens() const
    {
        const auto& dlens = data_ins_->get_shape().lens();
        return std::vector<std::size_t>(dlens.begin() + axis_index_ + 1, dlens.end());
    }

    std::vector<std::size_t> rest_lens() const
    {
        auto result = pre_lens();
        auto post   = post_lens();
        result.insert(result.end(), post.begin(), post.end());
        return result;
    }

    std::vector<std::size_t> idims() const
    {
        auto result = indices_arg_.get_shape().lens();
        assert(not result.empty() && "idims() returned empty vector!");
        return result;
    }

    std::vector<std::size_t> index_positions() const
    {
        std::vector<std::size_t> positions;
        const auto dims = idims();
        for(std::size_t i = 0; i < dims.size(); ++i)
        {
            if(dims[i] > 1)
                positions.push_back(i);
        }
        return positions;
    }

    std::vector<std::size_t> index_dims() const
    {
        std::vector<std::size_t> dims;
        const auto all_dims = idims();
        for(auto d : all_dims)
        {
            if(d > 1)
                dims.push_back(d);
        }
        return dims;
    }

    std::vector<std::size_t> target_shape() const
    {
        auto result = pre_lens();
        auto dims   = idims();
        assert(not dims.empty() && "idims() is empty in target_shape!");
        result.insert(result.end(), dims.begin(), dims.end());
        auto post = post_lens();
        result.insert(result.end(), post.begin(), post.end());
        assert(not result.empty() && "target_shape() returned empty vector!");
        return result;
    }

    // Lazy-loaded factor candidates with caching
    const std::vector<std::vector<std::size_t>>& factor_candidates() const
    {
        if(not factor_candidates_)
        {
            factor_candidates_ = std::make_shared<std::vector<std::vector<std::size_t>>>();
            constexpr std::size_t max_factorizations = 256;
            auto raw_factors = enumerate_all_factorizations(axis_len(), max_factorizations);

            for(auto& factors : raw_factors)
            {
                if(factor_candidates_->size() >= max_factorizations)
                    break;
                add_unique_factorization(
                    *factor_candidates_, std::move(factors), axis_len(), max_factorizations);
            }

            // Add factorizations from reshape chain if applicable
            const auto& dlens = data_ins_->get_shape().lens();
            if(dlens.size() == 1 and axis_index_ == 0)
            {
                instruction_ref curr_data = data_ins_;
                while(curr_data->name() == "reshape" and curr_data->inputs().size() == 1)
                {
                    curr_data              = curr_data->inputs().front();
                    const auto& input_lens = curr_data->get_shape().lens();
                    if(input_lens.empty())
                        break;
                    auto product = std::accumulate(
                        input_lens.begin(), input_lens.end(), std::size_t{1}, std::multiplies<>{});
                    if(product != axis_len())
                        break;
                    if(factor_candidates_->size() >= max_factorizations)
                        break;
                    std::vector<std::size_t> factors(input_lens.begin(), input_lens.end());
                    add_unique_factorization(
                        *factor_candidates_, std::move(factors), axis_len(), max_factorizations);
                }
            }
        }
        return *factor_candidates_;
    }

    // Mutable version for direct assignment (needed for ctx_1d case)
    void set_factor_candidates(std::shared_ptr<std::vector<std::vector<std::size_t>>> candidates)
    {
        factor_candidates_ = std::move(candidates);
    }

    // Factory method to create a context with reshaped indices (for 1D normalization or segments)
    static gather_context with_reshaped_indices(const gather_context& base,
                                                const std::vector<std::size_t>& new_indices_shape)
    {
        // Reshape the indices argument to the new shape (keeps same underlying data)
        shape new_shape{shape::int64_type, new_indices_shape};
        argument reshaped_indices = base.indices_arg_.reshape(new_shape);

        gather_context new_ctx(base.data_ins(), base.axis_index(), std::move(reshaped_indices));

        // Share the factor_candidates cache only if it was already initialized
        if(base.factor_candidates_)
        {
            new_ctx.set_factor_candidates(base.factor_candidates_);
        }
        return new_ctx;
    }
};

} // namespace

// ============================================================================
// Segment-Based Gather Optimization
// ============================================================================

namespace {

/// Metadata for constant segment
struct constant_segment_meta
{
    int64_t value;

    /// Detect constant segment pattern
    static std::optional<constant_segment_meta>
    detect(const std::vector<int64_t>& indices, std::size_t start, std::size_t length)
    {
        if(length == 0)
            return std::nullopt;
        auto value = indices[start];
        for(std::size_t i = start + 1; i < start + length; ++i)
        {
            if(indices[i] != value)
                return std::nullopt;
        }
        return constant_segment_meta{value};
    }

    /// Transform constant segment into instructions
    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        auto di = ctx.data_ins();
        auto rl = ctx.rest_lens();

        auto moved  = builder.move_axis_to_front(di, ctx.axis_index());
        auto sliced = builder.slice(moved, {0}, {value}, {value + 1});

        // Reshape to remove the sliced 1-dimension, giving us rest_lens shape
        instruction_ref reshaped = sliced;
        if(not rl.empty())
        {
            std::vector<int64_t> rest_shape(rl.begin(), rl.end());
            reshaped = builder.reshape(sliced, rest_shape);
        }

        // Insert a 1-dimension at the axis position for broadcasting
        std::vector<int64_t> with_axis_dim = to_int64_vec(ctx.pre_lens());
        with_axis_dim.push_back(1);
        auto post = ctx.post_lens(); // Store the result to ensure it lives long enough
        with_axis_dim.insert(with_axis_dim.end(), post.begin(), post.end());
        auto with_dim = builder.reshape(reshaped, with_axis_dim);

        // Now match_shape will broadcast the 1 to the index count
        auto target = ctx.target_shape();
        return builder.match_shape(with_dim, target);
    }
};

/// Metadata for contiguous segment
struct contiguous_segment_meta
{
    int64_t start;
    int64_t count;

    /// Detect contiguous segment pattern
    static std::optional<contiguous_segment_meta>
    detect(const std::vector<int64_t>& indices, std::size_t start, std::size_t length)
    {
        if(length == 0)
            return std::nullopt;
        auto first = indices[start];
        for(std::size_t i = 1; i < length; ++i)
        {
            if(indices[start + i] != first + static_cast<int64_t>(i))
                return std::nullopt;
        }
        return contiguous_segment_meta{first, static_cast<int64_t>(length)};
    }

    /// Transform contiguous segment into instructions
    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        auto moved  = builder.move_axis_to_front(ctx.data_ins(), ctx.axis_index());
        auto sliced = builder.slice(moved, {0}, {start}, {start + count});
        auto restored =
            builder.restore_axis_position(sliced, ctx.pre_lens().size(), 1, ctx.post_lens().size());
        return builder.match_shape(restored, ctx.target_shape());
    }
};

/// Metadata for arithmetic segment
struct arithmetic_segment_meta
{
    int64_t base;
    int64_t stride;
    std::size_t count;

    /// Detect arithmetic segment pattern
    static std::optional<arithmetic_segment_meta>
    detect(const std::vector<int64_t>& indices, std::size_t start, std::size_t length)
    {
        if(length < 2)
            return std::nullopt;
        auto base   = indices[start];
        auto stride = indices[start + 1] - base;
        if(stride <= 1 or base < 0 or base >= stride)
            return std::nullopt;
        for(std::size_t i = 0; i < length; ++i)
        {
            if(indices[start + i] != base + static_cast<int64_t>(i) * stride)
                return std::nullopt;
        }
        return arithmetic_segment_meta{base, stride, length};
    }

    /// Transform arithmetic segment into instructions
    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        auto moved = builder.move_axis_to_front(ctx.data_ins(), ctx.axis_index());

        // For arithmetic patterns: indices = base + k*stride for k in [0, count)
        // We need to extract every stride-th element starting from base
        // Use slice + step: start=base, end=base+count*stride, step=stride
        auto max_index = base + static_cast<int64_t>(count) * stride;
        auto sliced    = builder.slice_with_step(moved, {0}, {base}, {max_index}, {stride});

        // After slice + step with stride, we have exactly `count` elements along axis 0
        // Reshape to final dimensions
        std::vector<int64_t> final_dims = {static_cast<int64_t>(count)};
        auto rest                       = ctx.rest_lens(); // Store to ensure lifetime
        final_dims.insert(final_dims.end(), rest.begin(), rest.end());
        auto reshaped = builder.reshape(sliced, final_dims);

        auto restored = builder.restore_axis_position(
            reshaped, ctx.pre_lens().size(), 1, ctx.post_lens().size());
        return builder.match_shape(restored, ctx.target_shape());
    }
};

/// Metadata for RTR window segment
struct rtr_window_segment_meta
{
    std::vector<std::size_t> factors;
    std::vector<std::size_t> permutation;

    /// Check if indices form valid permutation
    static bool is_valid_permutation_seg(const std::vector<int64_t>& indices,
                                         std::size_t start,
                                         std::size_t length)
    {
        if(length == 0)
            return false;
        std::set<int64_t> seen;
        for(std::size_t i = start; i < start + length; ++i)
        {
            auto val = indices[i];
            if(val < 0 or static_cast<std::size_t>(val) >= length)
                return false;
            if(seen.count(val) > 0)
                return false;
            seen.insert(val);
        }
        return true;
    }

    /// Try grid factorization
    static bool try_grid_factorization_seg(const std::vector<int64_t>& indices,
                                           std::size_t start,
                                           std::size_t length,
                                           const std::vector<std::size_t>& factors,
                                           std::vector<std::size_t>& out_permutation)
    {
        if(product_of(factors) != length)
            return false;

        std::vector<std::vector<std::size_t>> multi_indices(length);
        for(std::size_t i = 0; i < length; ++i)
        {
            auto idx = static_cast<std::size_t>(indices[start + i]);
            if(idx >= length)
                return false;
            auto temp = idx;
            multi_indices[i].resize(factors.size());
            for(int j = static_cast<int>(factors.size()) - 1; j >= 0; --j)
            {
                multi_indices[i][j] = temp % factors[j];
                temp /= factors[j];
            }
        }

        if(factors.size() > 4)
            return false;

        std::vector<std::size_t> perm(factors.size());
        std::iota(perm.begin(), perm.end(), std::size_t{0});

        do
        {
            bool valid = true;
            for(std::size_t i = 0; i < length and valid; ++i)
            {
                std::size_t expected = 0;
                std::size_t stride   = 1;
                for(int j = static_cast<int>(factors.size()) - 1; j >= 0; --j)
                {
                    expected += multi_indices[i][perm[j]] * stride;
                    stride *= factors[perm[j]];
                }
                if(expected != i)
                    valid = false;
            }
            if(valid)
            {
                out_permutation = perm;
                return true;
            }
        } while(std::next_permutation(perm.begin(), perm.end()));

        return false;
    }

    /// Detect RTR window segment pattern
    static std::optional<rtr_window_segment_meta>
    detect(const std::vector<int64_t>& indices,
           std::size_t start,
           std::size_t length,
           const std::vector<std::vector<std::size_t>>& factor_candidates)
    {
        if(not is_valid_permutation_seg(indices, start, length))
            return std::nullopt;
        for(const auto& factors : factor_candidates)
        {
            if(product_of(factors) != length)
                continue;
            std::vector<std::size_t> permutation;
            if(try_grid_factorization_seg(indices, start, length, factors, permutation))
                return rtr_window_segment_meta{factors, permutation};
        }
        // Don't return identity RTR - let other patterns match instead
        return std::nullopt;
    }

    /// Transform RTR window segment into instructions
    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        auto moved = builder.move_axis_to_front(ctx.data_ins(), ctx.axis_index());
        std::vector<int64_t> reshape_dims;
        std::transform(factors.begin(),
                       factors.end(),
                       std::back_inserter(reshape_dims),
                       [](auto f) { return static_cast<int64_t>(f); });
        auto rest = ctx.rest_lens(); // Store to ensure lifetime
        reshape_dims.insert(reshape_dims.end(), rest.begin(), rest.end());
        auto reshaped = builder.reshape(moved, reshape_dims);

        std::vector<int64_t> full_perm;
        std::transform(permutation.begin(),
                       permutation.end(),
                       std::back_inserter(full_perm),
                       [](auto p) { return static_cast<int64_t>(p); });
        for(std::size_t i = factors.size(); i < reshape_dims.size(); ++i)
            full_perm.push_back(static_cast<int64_t>(i));

        auto transposed                 = builder.transpose(reshaped, full_perm);
        std::vector<int64_t> final_dims = {static_cast<int64_t>(
            std::accumulate(factors.begin(), factors.end(), std::size_t{1}, std::multiplies<>{}))};
        final_dims.insert(final_dims.end(), rest.begin(), rest.end()); // Reuse 'rest' from above
        auto final_reshape = builder.reshape(transposed, final_dims);
        auto restored      = builder.restore_axis_position(
            final_reshape, ctx.pre_lens().size(), 1, ctx.post_lens().size());
        return builder.match_shape(restored, ctx.target_shape());
    }
};

/// Index segment with pattern metadata
struct index_segment
{
    using meta_type       = std::variant<std::monostate,
                                         constant_segment_meta,
                                         contiguous_segment_meta,
                                         arithmetic_segment_meta,
                                         rtr_window_segment_meta>;
    std::size_t start_pos = 0;
    std::size_t length    = 0;
    meta_type metadata    = std::monostate{};

    template <class T>
    bool has_type() const
    {
        return std::holds_alternative<T>(metadata);
    }

    bool empty() const { return std::holds_alternative<std::monostate>(metadata); }

    /// Get representative value for this segment (used for level-2 analysis)
    /// Returns the "start" or "min" value that characterizes the segment
    std::optional<int64_t> representative_value() const
    {
        return std::visit(
            [](const auto& m) -> std::optional<int64_t> {
                using T = std::decay_t<decltype(m)>;
                if constexpr(std::is_same_v<T, constant_segment_meta>)
                    return m.value;
                else if constexpr(std::is_same_v<T, contiguous_segment_meta>)
                    return m.start;
                else if constexpr(std::is_same_v<T, arithmetic_segment_meta>)
                    return m.base; // arithmetic uses 'base' not 'start'
                else if constexpr(std::is_same_v<T, rtr_window_segment_meta>)
                    return std::nullopt; // RTR doesn't have a simple representative
                else
                    return std::nullopt;
            },
            metadata);
    }

    static index_segment detect(const std::vector<int64_t>& indices,
                                std::size_t pos,
                                std::size_t len,
                                const std::vector<std::vector<std::size_t>>& factor_candidates)
    {
        if(auto meta = constant_segment_meta::detect(indices, pos, len))
        {
            return index_segment{pos, len, *meta};
        }
        else if(auto meta_cont = contiguous_segment_meta::detect(indices, pos, len))
        {
            return index_segment{pos, len, *meta_cont};
        }
        else if(auto meta_arith = arithmetic_segment_meta::detect(indices, pos, len))
        {
            return index_segment{pos, len, *meta_arith};
        }
        else if(auto meta_rtr =
                    rtr_window_segment_meta::detect(indices, pos, len, factor_candidates))
        {
            return index_segment{pos, len, *meta_rtr};
        }
        return {};
    }

    static index_segment
    find_first_segment(const std::vector<int64_t>& indices,
                       const std::vector<std::vector<std::size_t>>& factor_candidates)
    {
        std::size_t n = 2 + indices.size() / 2;
        for(auto i : range(1, n))
        {
            if(indices.size() % i != 0)
                continue;
            auto seg = detect(indices, 0, indices.size() / i, factor_candidates);
            if(not seg.empty())
                return seg;
        }
        return {};
    }

    /// Analyze indices into segments
    static std::vector<index_segment>
    analyze(const std::vector<int64_t>& indices,
            std::size_t /* axis_len */,
            const std::vector<std::vector<std::size_t>>& factor_candidates)
    {
        std::vector<index_segment> segments;
        if(indices.empty())
            return segments;
        segments.push_back(find_first_segment(indices, factor_candidates));
        if(segments.front().empty())
            return {};
        transform(range(1, indices.size() / segments[0].length),
                  std::back_inserter(segments),
                  [&](auto i) {
                      return detect(
                          indices, i * segments[0].length, segments[0].length, factor_candidates);
                  });
        if(std::all_of(
               segments.begin(), segments.end(), [](const auto& s) { return not s.empty(); }))
            return segments;
        return {};
    }
};

static std::vector<std::size_t> make_segment_target_shape(const gather_context& ctx,
                                                          std::size_t segment_length)
{
    assert(segment_length != 0);
    std::vector<std::size_t> result = ctx.pre_lens();
    result.push_back(segment_length);
    auto post = ctx.post_lens(); // Store to ensure lifetime
    result.insert(result.end(), post.begin(), post.end());
    return result;
}

static instruction_ref apply_segment_transform(const index_segment& segment,
                                               const gather_context& ctx,
                                               gather_instruction_builder& builder)
{
    assert(not segment.empty());
    auto ensure_shape = [&](instruction_ref result) {
        assert(result != instruction_ref{});
        assert(result->get_shape().lens() == ctx.target_shape());
        return result;
    };
    if(segment.has_type<constant_segment_meta>())
        return ensure_shape(
            std::get<constant_segment_meta>(segment.metadata).transform(ctx, builder));
    if(segment.has_type<contiguous_segment_meta>())
        return ensure_shape(
            std::get<contiguous_segment_meta>(segment.metadata).transform(ctx, builder));
    if(segment.has_type<arithmetic_segment_meta>())
        return ensure_shape(
            std::get<arithmetic_segment_meta>(segment.metadata).transform(ctx, builder));
    if(segment.has_type<rtr_window_segment_meta>())
        return ensure_shape(
            std::get<rtr_window_segment_meta>(segment.metadata).transform(ctx, builder));
    assert(false && "Unsupported segment type for transform");
    return instruction_ref{};
}

/// Pattern: 2-way split
struct split_pattern
{
    std::array<index_segment, 2> segments;

    /// Detect split pattern (2-way only)
    static std::optional<split_pattern> detect(const std::vector<index_segment>& segments_vec)
    {
        if(segments_vec.size() != 2)
            return std::nullopt;
        const auto& first  = segments_vec[0];
        const auto& second = segments_vec[1];
        if(first.empty() or second.empty())
            return std::nullopt;
        auto first_index  = first.metadata.index();
        auto second_index = second.metadata.index();
        if(first_index == 0 or second_index == 0)
            return std::nullopt;

        if(first_index == second_index)
        {
            if(first.has_type<contiguous_segment_meta>() and
               second.has_type<contiguous_segment_meta>())
            {
                const auto& first_meta  = std::get<contiguous_segment_meta>(first.metadata);
                const auto& second_meta = std::get<contiguous_segment_meta>(second.metadata);
                if(first_meta.start != second_meta.start)
                    return split_pattern{{first, second}};
            }
            return std::nullopt;
        }
        return split_pattern{{first, second}};
    }

    /// Transform split pattern into instructions
    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        std::vector<instruction_ref> parts;
        parts.reserve(segments.size());
        for(const auto& segment : segments)
        {
            // Create a temporary context with different indices shape for this segment
            auto segment_indices_shape = std::vector<std::size_t>{segment.length};
            auto segment_ctx = gather_context::with_reshaped_indices(ctx, segment_indices_shape);
            parts.push_back(apply_segment_transform(segment, segment_ctx, builder));
        }
        auto axis         = static_cast<int64_t>(ctx.pre_lens().size());
        auto concatenated = builder.concat(parts, axis);
        return builder.match_shape(concatenated, ctx.target_shape());
    }
};

// ============================================================================
// Composable 2D Segment Optimization [N segments × S length]
// ============================================================================

/// Describes how N segments vary from each other
struct inter_segment_variation
{
    enum class pattern_type
    {
        identical,    // All segments have identical metadata
        contiguous,   // Segments form contiguous sequence
        arithmetic,   // Segments follow arithmetic pattern (for arithmetic segments)
        heterogeneous // Different patterns (use split or custom)
    };

    pattern_type type  = pattern_type::heterogeneous;
    std::size_t stride = 0; // For arithmetic pattern

    /// Detect variation pattern for segments of the same type
    template <class SegmentMeta>
    static std::optional<inter_segment_variation>
    detect_for_type(const std::vector<index_segment>& segments)
    {
        if(segments.empty() or not segments.front().has_type<SegmentMeta>())
            return std::nullopt;

        // Check all segments have the same type
        if(not std::all_of(segments.begin(), segments.end(), [](const auto& seg) {
               return seg.template has_type<SegmentMeta>();
           }))
            return std::nullopt;

        // Check all have same length
        auto first_length = segments.front().length;
        if(not std::all_of(segments.begin(), segments.end(), [first_length](const auto& seg) {
               return seg.length == first_length;
           }))
            return std::nullopt;

        if constexpr(std::is_same_v<SegmentMeta, constant_segment_meta>)
        {
            // Check if all constants are identical
            auto first_value = std::get<constant_segment_meta>(segments.front().metadata).value;
            bool all_same =
                std::all_of(segments.begin(), segments.end(), [first_value](const auto& seg) {
                    return std::get<constant_segment_meta>(seg.metadata).value == first_value;
                });
            if(all_same)
                return inter_segment_variation{pattern_type::identical, 0};

            // Otherwise heterogeneous
            return inter_segment_variation{pattern_type::heterogeneous, 0};
        }
        else if constexpr(std::is_same_v<SegmentMeta, contiguous_segment_meta>)
        {
            // Check if segments form a contiguous sequence
            auto first_meta    = std::get<contiguous_segment_meta>(segments.front().metadata);
            bool is_contiguous = true;
            for(std::size_t i = 1; i < segments.size() and is_contiguous; ++i)
            {
                auto meta      = std::get<contiguous_segment_meta>(segments[i].metadata);
                auto prev_meta = std::get<contiguous_segment_meta>(segments[i - 1].metadata);
                is_contiguous  = (meta.start == prev_meta.start + prev_meta.count and
                                 meta.count == first_meta.count);
            }
            if(is_contiguous)
                return inter_segment_variation{pattern_type::contiguous, 0};

            return inter_segment_variation{pattern_type::heterogeneous, 0};
        }
        else if constexpr(std::is_same_v<SegmentMeta, arithmetic_segment_meta>)
        {
            // Check if all arithmetic segments have same stride/count, bases are 0,1,2,...
            auto first_meta = std::get<arithmetic_segment_meta>(segments.front().metadata);
            auto stride     = first_meta.stride;
            auto count      = first_meta.count;

            bool is_arithmetic_tiled =
                std::all_of(segments.begin(), segments.end(), [stride, count](const auto& seg) {
                    auto meta = std::get<arithmetic_segment_meta>(seg.metadata);
                    return meta.stride == stride and meta.count == count;
                });

            if(is_arithmetic_tiled)
            {
                // Check bases are 0, 1, 2, ...
                bool bases_sequential = true;
                for(std::size_t i = 0; i < segments.size() and bases_sequential; ++i)
                {
                    auto meta        = std::get<arithmetic_segment_meta>(segments[i].metadata);
                    bases_sequential = (meta.base == static_cast<int64_t>(i));
                }
                if(bases_sequential)
                    return inter_segment_variation{pattern_type::arithmetic,
                                                   static_cast<std::size_t>(stride)};
            }

            return inter_segment_variation{pattern_type::heterogeneous, 0};
        }
        else
        {
            // For rtr_window or other types, check if identical
            // For now, treat as heterogeneous
            return inter_segment_variation{pattern_type::heterogeneous, 0};
        }
    }

    /// Detect variation pattern for any segment collection
    static std::optional<inter_segment_variation> detect(const std::vector<index_segment>& segments)
    {
        if(segments.empty())
            return std::nullopt;

        // Try each segment type
        if(auto var = detect_for_type<constant_segment_meta>(segments))
            return var;
        if(auto var = detect_for_type<contiguous_segment_meta>(segments))
            return var;
        if(auto var = detect_for_type<arithmetic_segment_meta>(segments))
            return var;
        if(auto var = detect_for_type<rtr_window_segment_meta>(segments))
            return var;

        return std::nullopt;
    }

    /// Detect pattern in a sequence of representative values (for level-2 analysis)
    static std::optional<inter_segment_variation> detect_pattern(const std::vector<int64_t>& values)
    {
        if(values.empty())
            return std::nullopt;

        // Check if all identical
        bool all_same =
            std::all_of(values.begin(), values.end(), [&](int64_t v) { return v == values[0]; });
        if(all_same)
            return inter_segment_variation{pattern_type::identical, 0};

        // Check if contiguous
        bool is_contiguous = true;
        for(std::size_t i = 1; i < values.size(); ++i)
        {
            if(values[i] != values[i - 1] + 1)
            {
                is_contiguous = false;
                break;
            }
        }
        if(is_contiguous)
            return inter_segment_variation{pattern_type::contiguous, 1};

        // Check if arithmetic progression
        if(values.size() >= 2)
        {
            int64_t stride     = values[1] - values[0];
            bool is_arithmetic = true;
            for(std::size_t i = 2; i < values.size(); ++i)
            {
                if(values[i] - values[i - 1] != stride)
                {
                    is_arithmetic = false;
                    break;
                }
            }
            if(is_arithmetic && stride != 0)
                return inter_segment_variation{pattern_type::arithmetic,
                                               static_cast<std::size_t>(std::abs(stride))};
        }

        // Heterogeneous
        return inter_segment_variation{pattern_type::heterogeneous, 0};
    }
};

/// Transform repeated homogeneous segments composably as [N segments × S length]
struct repeated_segment_transformer
{
    std::vector<index_segment> segments;
    inter_segment_variation variation;

    static std::optional<repeated_segment_transformer>
    detect(const std::vector<index_segment>& segs)
    {
        if(segs.size() < 2)
            return std::nullopt;

        auto var = inter_segment_variation::detect(segs);
        if(not var.has_value())
            return std::nullopt;

        return repeated_segment_transformer{segs, *var};
    }

    instruction_ref transform(const gather_context& ctx, gather_instruction_builder& builder) const
    {
        auto N = segments.size();         // Number of segments
        auto S = segments.front().length; // Segment length

        // Step 1: Handle based on variation type
        if(variation.type == inter_segment_variation::pattern_type::identical)
        {
            // All segments identical - transform one and broadcast
            auto segment_ctx   = gather_context::with_reshaped_indices(ctx, {S});
            auto single_result = apply_segment_transform(segments.front(), segment_ctx, builder);

            // The single result has shape [pre_lens, S, post_lens]
            // We need [pre_lens, N*S, post_lens]
            // Insert 1-dim at axis position and broadcast
            auto pre_lens                   = ctx.pre_lens();
            std::vector<int64_t> with_n_dim = to_int64_vec(pre_lens);
            with_n_dim.push_back(1);
            with_n_dim.push_back(static_cast<int64_t>(S));
            auto post = ctx.post_lens();
            with_n_dim.insert(with_n_dim.end(), post.begin(), post.end());

            auto reshaped = builder.reshape(single_result, with_n_dim);

            // Broadcast the N dimension
            std::vector<int64_t> broadcast_shape = to_int64_vec(pre_lens);
            broadcast_shape.push_back(static_cast<int64_t>(N));
            broadcast_shape.push_back(static_cast<int64_t>(S));
            broadcast_shape.insert(broadcast_shape.end(), post.begin(), post.end());
            auto broadcasted = builder.multibroadcast(reshaped, broadcast_shape);

            // Flatten N*S
            std::vector<int64_t> final_shape = to_int64_vec(pre_lens);
            final_shape.push_back(static_cast<int64_t>(N * S));
            final_shape.insert(final_shape.end(), post.begin(), post.end());
            return builder.reshape(broadcasted, final_shape);
        }
        else if(variation.type == inter_segment_variation::pattern_type::contiguous)
        {
            // Segments form contiguous sequence - can optimize with single slice
            auto first_meta = std::get<contiguous_segment_meta>(segments.front().metadata);
            auto last_meta  = std::get<contiguous_segment_meta>(segments.back().metadata);

            auto moved = builder.move_axis_to_front(ctx.data_ins(), ctx.axis_index());
            auto sliced =
                builder.slice(moved, {0}, {first_meta.start}, {last_meta.start + last_meta.count});
            auto restored = builder.restore_axis_position(
                sliced, ctx.pre_lens().size(), 1, ctx.post_lens().size());
            return builder.match_shape(restored, ctx.target_shape());
        }
        else if(variation.type == inter_segment_variation::pattern_type::arithmetic)
        {
            // Arithmetic tiling: reshape data to [stride, S] and transpose
            auto stride = variation.stride;
            auto moved  = builder.move_axis_to_front(ctx.data_ins(), ctx.axis_index());
            auto rest   = ctx.rest_lens();

            std::vector<int64_t> reshape_dims = {static_cast<int64_t>(stride),
                                                 static_cast<int64_t>(S)};
            reshape_dims.insert(reshape_dims.end(), rest.begin(), rest.end());
            auto reshaped = builder.reshape(moved, reshape_dims);

            // Transpose [stride, S, ...rest] → [S, stride, ...rest]
            std::vector<int64_t> perm = {1, 0};
            for(std::size_t i = 2; i < reshape_dims.size(); ++i)
                perm.push_back(static_cast<int64_t>(i));
            auto transposed = builder.transpose(reshaped, perm);

            // Flatten S*stride
            std::vector<int64_t> final_dims = {static_cast<int64_t>(S * stride)};
            final_dims.insert(final_dims.end(), rest.begin(), rest.end());
            auto final_reshape = builder.reshape(transposed, final_dims);

            auto restored = builder.restore_axis_position(
                final_reshape, ctx.pre_lens().size(), 1, ctx.post_lens().size());
            return builder.match_shape(restored, ctx.target_shape());
        }
        else
        {
            // Heterogeneous - fall back to split or concat approach
            return instruction_ref{};
        }
    }
};

/// 2-level hierarchical segment transformer
/// Analyzes both intra-segment patterns (level 1) and inter-segment patterns (level 2)
struct two_level_segment_transformer
{
    std::vector<index_segment> segments;
    std::vector<int64_t> representatives;     // Representative values from each segment
    inter_segment_variation level2_variation; // Pattern of the representatives

    /// Detect 2-level pattern from segments
    static std::optional<two_level_segment_transformer>
    detect(const std::vector<index_segment>& segs)
    {
        if(segs.empty() || segs.size() == 1)
            return std::nullopt;

        // Check all segments have the same length (level 1 homogeneity)
        std::size_t seg_length = segs[0].length;
        for(const auto& seg : segs)
        {
            if(seg.length != seg_length)
                return std::nullopt;
        }

        // Extract representative values from each segment
        std::vector<int64_t> reps;
        reps.reserve(segs.size());
        for(const auto& seg : segs)
        {
            auto rep = seg.representative_value();
            if(not rep.has_value())
                return std::nullopt; // Can't analyze segments without representatives
            reps.push_back(*rep);
        }

        // Analyze the pattern of representatives (level 2)
        // Try to detect pattern in the representative values
        auto level2_var = inter_segment_variation::detect_pattern(reps);
        if(not level2_var.has_value())
            return std::nullopt;

        return two_level_segment_transformer{segs, reps, *level2_var};
    }

    /// Transform using 2-level analysis on 1D flattened data
    instruction_ref transform(const gather_context& ctx,
                              gather_instruction_builder& builder,
                              const std::vector<std::size_t>& original_target_shape) const
    {
        // Dispatch based on level-1 segment type
        // All segments must have the same type (checked in detect())
        if(segments[0].template has_type<constant_segment_meta>())
        {
            return transform_constant_segments(ctx, builder, original_target_shape);
        }
        else if(segments[0].template has_type<contiguous_segment_meta>())
        {
            return transform_contiguous_segments(ctx, builder, original_target_shape);
        }
        else if(segments[0].template has_type<arithmetic_segment_meta>())
        {
            return transform_arithmetic_segments(ctx, builder, original_target_shape);
        }

        return instruction_ref{};
    }

    private:
    /// Transform pattern with all constant segments (resize/broadcast pattern)
    instruction_ref transform_constant_segments(const gather_context& ctx,
                                                gather_instruction_builder& builder,
                                                const std::vector<std::size_t>& target_shape) const
    {
        // Constant segments: each segment accesses the same value repeatedly
        // This is a broadcast/resize pattern in 1D
        // Goal: gather the unique values, then broadcast to output shape

        auto seg_len   = segments.front().length;
        auto num_segs  = segments.size();
        auto flat_data = ctx.data_ins();

        // Extract the constant values from each segment
        std::vector<int64_t> values;
        values.reserve(num_segs);
        for(const auto& seg : segments)
        {
            const auto& meta = std::get<constant_segment_meta>(seg.metadata);
            values.push_back(meta.value);
        }

        // Check how many unique values are accessed
        std::set<int64_t> unique_values(values.begin(), values.end());
        std::size_t num_unique = unique_values.size();

        // Simple approach: use repeated_segment_transformer logic
        // This pattern is similar to identical segments but with different constant values
        // Fall back to letting the gather execute - this transformation is complex
        // without knowing the original multi-dimensional structure

        return instruction_ref{};
    }

    /// Transform pattern with all contiguous segments (multi-axis stride pattern)
    instruction_ref
    transform_contiguous_segments(const gather_context& ctx,
                                  gather_instruction_builder& builder,
                                  const std::vector<std::size_t>& target_shape) const
    {
        // Multi-axis stride: N contiguous segments of length L accessing 1D flattened data
        // Pattern: starts tell us how segments are distributed
        // Goal: Factor the 1D access into N×L structure, then reshape/slice/transpose

        auto seg_len           = segments.front().length;
        auto num_segs          = segments.size();
        auto flat_data         = ctx.data_ins();
        std::size_t total_size = ctx.axis_len();

        // Extract starts from contiguous segments
        std::vector<int64_t> starts;
        starts.reserve(num_segs);
        for(const auto& seg : segments)
        {
            const auto& meta = std::get<contiguous_segment_meta>(seg.metadata);
            starts.push_back(meta.start);
        }

        // Analyze starts to find the smallest stride (GCD of all differences)
        std::vector<int64_t> sorted_starts = starts;
        std::sort(sorted_starts.begin(), sorted_starts.end());

        std::vector<int64_t> diffs;
        for(std::size_t i = 1; i < sorted_starts.size(); ++i)
        {
            auto diff = sorted_starts[i] - sorted_starts[i - 1];
            if(diff > 0)
                diffs.push_back(diff);
        }

        if(diffs.empty())
            return instruction_ref{};

        int64_t min_stride = *std::min_element(diffs.begin(), diffs.end());

        // Determine number of interleaved sequences
        // Count unique offsets modulo min_stride
        std::set<int64_t> unique_offsets;
        for(auto s : sorted_starts)
            unique_offsets.insert(s % min_stride);

        std::size_t num_interleaved = unique_offsets.size();

        // Factorize: total_size = F1 × num_interleaved × min_stride / (min_stride / seg_len) ×
        // seg_len Simplify based on pattern: we want reshape to [F1, num_interleaved, F2, seg_len]
        // where F1 * num_interleaved * F2 * seg_len = total_size

        if(total_size % (num_interleaved * seg_len) != 0)
            return instruction_ref{};

        std::size_t remaining = total_size / (num_interleaved * seg_len);

        // Try factoring remaining into F1 × F2 where F2 relates to the stride pattern
        std::size_t f2 = static_cast<std::size_t>(min_stride) / seg_len;
        if(f2 == 0)
            f2 = 1;

        if(remaining % f2 != 0)
            return instruction_ref{};

        std::size_t f1 = remaining / f2;

        // Reshape to [f1, num_interleaved, f2, seg_len]
        std::vector<int64_t> reshape_dims = {static_cast<int64_t>(f1),
                                             static_cast<int64_t>(num_interleaved),
                                             static_cast<int64_t>(f2),
                                             static_cast<int64_t>(seg_len)};

        auto reshaped = builder.reshape(flat_data, reshape_dims);

        // Determine if we need to slice
        std::size_t num_output_elems = num_segs * seg_len;
        if(num_output_elems < total_size)
        {
            // Calculate slice bounds from the access pattern
            // We're extracting num_segs out of possible (f1 * num_interleaved * f2) segment
            // positions
            std::size_t total_seg_positions = f1 * num_interleaved * f2;

            // Slice dimensions that are being subsampled
            std::vector<int64_t> slice_axes;
            std::vector<int64_t> slice_starts;
            std::vector<int64_t> slice_ends;

            // Determine which dimensions to slice based on coverage
            for(std::size_t dim = 0; dim < 3; ++dim)
            {
                std::size_t dim_size    = static_cast<std::size_t>(reshape_dims[dim]);
                std::size_t dim_covered = dim_size; // Default: use full dimension

                // Check if this dimension needs slicing
                if(dim == 1 && num_segs < total_seg_positions)
                {
                    dim_covered = std::min(num_segs / (f1 * f2), dim_size);
                    if(dim_covered < dim_size)
                    {
                        slice_axes.push_back(static_cast<int64_t>(dim));
                        slice_starts.push_back(0);
                        slice_ends.push_back(static_cast<int64_t>(dim_covered));
                    }
                }
                else if(dim == 2 && num_output_elems < total_size)
                {
                    std::size_t seg_groups = total_seg_positions / f2;
                    if(num_segs <= seg_groups)
                    {
                        dim_covered = (num_segs * f2) / (f1 * num_interleaved);
                        if(dim_covered < dim_size)
                        {
                            slice_axes.push_back(static_cast<int64_t>(dim));
                            slice_starts.push_back(0);
                            slice_ends.push_back(static_cast<int64_t>(dim_covered));
                        }
                    }
                }
            }

            if(not slice_axes.empty())
                reshaped = builder.slice(reshaped, slice_axes, slice_starts, slice_ends);
        }

        // Transpose to match gather output order
        // The segments appear in a specific order in the gather output
        // We need to move the seg_len dimension and reorder based on how segments were accessed
        std::vector<int64_t> perm = {2, 0, 1, 3}; // Move f2 to front, keep order
        reshaped                  = builder.transpose(reshaped, perm);

        return builder.match_shape(reshaped, target_shape);
    }

    /// Transform pattern with all arithmetic segments (channel patch pattern)
    instruction_ref
    transform_arithmetic_segments(const gather_context& ctx,
                                  gather_instruction_builder& builder,
                                  const std::vector<std::size_t>& target_shape) const
    {
        // Arithmetic segments: N segments with arithmetic pattern (base, base+stride,
        // base+2*stride, ...) Goal: Factor the 1D access into a structure that can be
        // reshaped/sliced/transposed

        const auto& first_meta = std::get<arithmetic_segment_meta>(segments.front().metadata);
        auto stride            = first_meta.stride;
        auto count             = first_meta.count;
        auto num_segs          = segments.size();

        // Verify all segments have same stride and count
        for(const auto& seg : segments)
        {
            const auto& meta = std::get<arithmetic_segment_meta>(seg.metadata);
            if(meta.stride != stride or meta.count != count)
                return instruction_ref{};
        }

        // Extract bases from segments
        std::vector<int64_t> bases;
        bases.reserve(num_segs);
        for(const auto& seg : segments)
        {
            const auto& meta = std::get<arithmetic_segment_meta>(seg.metadata);
            bases.push_back(meta.base);
        }

        auto flat_data         = ctx.data_ins();
        std::size_t total_size = ctx.axis_len();

        // The stride tells us the periodicity in the 1D array
        // Bases tell us which positions within the first period are accessed
        // Pattern: reshape to expose [period_count, stride, ...], then slice/transpose

        // Check if stride divides total_size evenly
        if(total_size % static_cast<std::size_t>(stride) != 0)
            return instruction_ref{};

        std::size_t period_count = total_size / static_cast<std::size_t>(stride);

        // We're accessing 'count' elements from each period at specific base offsets
        // Reshape to [count, period_count, stride / count, factors_of_bases...]

        // Find min/max bases to determine the slice region within each period
        int64_t min_base = *std::min_element(bases.begin(), bases.end());
        int64_t max_base = *std::max_element(bases.begin(), bases.end());

        // Bases span from min_base to max_base within the stride
        std::size_t base_span = static_cast<std::size_t>(max_base - min_base) + 1;

        // Try to factor the base span into dimensions
        // For bases [5,9,6,10], span = 6, factor as 2×3 or 3×2
        std::vector<std::size_t> base_factors;
        for(std::size_t f = 2; f * f <= base_span; ++f)
        {
            if(base_span % f == 0)
            {
                base_factors.push_back(f);
                base_factors.push_back(base_span / f);
                break;
            }
        }

        if(base_factors.empty())
            base_factors = {1, base_span};

        // Reshape to expose structure: [period_count, stride]
        // Then slice to extract accessed elements
        // Then reshape/transpose to target

        std::vector<int64_t> reshape_dims = {static_cast<int64_t>(period_count),
                                             static_cast<int64_t>(stride)};

        auto reshaped = builder.reshape(flat_data, reshape_dims);

        // Slice to extract the base span
        std::vector<int64_t> slice_axes   = {1};
        std::vector<int64_t> slice_starts = {min_base};
        std::vector<int64_t> slice_ends   = {min_base + static_cast<int64_t>(base_span)};
        auto sliced = builder.slice(reshaped, slice_axes, slice_starts, slice_ends);

        // Now we have [period_count, base_span]
        // Reshape to [period_count, f1, f2] where f1 * f2 = base_span
        std::vector<int64_t> expanded_dims = {static_cast<int64_t>(period_count),
                                              static_cast<int64_t>(base_factors[0]),
                                              static_cast<int64_t>(base_factors[1])};
        auto expanded                      = builder.reshape(sliced, expanded_dims);

        // Unsqueeze to separate dimensions for transpose
        std::vector<int64_t> unsqueeze_axes = {1, 2};
        auto unsqueezed                     = builder.unsqueeze(expanded, unsqueeze_axes);

        // Transpose to reorder dimensions
        // Result has shape [period_count, 1, base_factors[0], 1, base_factors[1]]
        // Transpose to move factors to front
        std::size_t ndim          = 2 + unsqueeze_axes.size() + 2; // period + unsq + factors
        std::vector<int64_t> perm = {4, 2, 0, 1, 3};               // Move last two to front
        while(perm.size() < ndim)
            perm.push_back(static_cast<int64_t>(perm.size()));

        auto transposed = builder.transpose(unsqueezed, perm);

        // Reshape to target
        return builder.match_shape(transposed, target_shape);
    }
};

/// Try segment-based optimization (assumes 1D indices in context)
/// Returns the optimized instruction if successful, nullopt otherwise
inline std::optional<instruction_ref>
try_segment_based_optimization_1d(const gather_context& ctx,
                                  gather_instruction_builder& builder,
                                  const std::vector<std::size_t>& original_target_shape)
{
    auto segments =
        index_segment::analyze(ctx.indices_values(), ctx.axis_len(), ctx.factor_candidates());

    if(segments.empty())
    {
        // No recognizable segments - cannot optimize
        return std::nullopt;
    }

    // Try single-segment patterns
    if(segments.size() == 1)
    {
        return std::visit(
            [&](const auto& m) -> std::optional<instruction_ref> {
                if constexpr(not std::is_same<std::decay_t<decltype(m)>, std::monostate>{})
                {
                    return m.transform(ctx, builder);
                }
                else
                {
                    return std::nullopt;
                }
            },
            segments[0].metadata);
    }

    // Try composable repeated segment pattern (handles homogeneous segments)
    if(auto repeated = repeated_segment_transformer::detect(segments))
    {
        auto result = repeated->transform(ctx, builder);
        if(result != instruction_ref{})
            return result;
    }

    // Try split pattern (2-way heterogeneous split)
    if(auto split = split_pattern::detect(segments))
    {
        return split->transform(ctx, builder);
    }

    // Try 2-level hierarchical pattern (when segments form a 2D structure)
    if(auto two_level = two_level_segment_transformer::detect(segments))
    {
        auto result = two_level->transform(ctx, builder, original_target_shape);
        if(result != instruction_ref{})
            return result;
    }

    // No more patterns to try
    return std::nullopt;
} /// Try segment-based optimization with multi-dimensional normalization
inline bool try_segment_based_optimization(module& m,
                                           instruction_ref ins,
                                           const gather_context& ctx,
                                           gather_instruction_builder& builder)
{
    // For 1D or scalar indices, use direct optimization
    if(ctx.idims().size() <= 1)
    {
        auto result = try_segment_based_optimization_1d(ctx, builder, ctx.target_shape());
        if(not result.has_value())
            return false;

        m.replace_instruction(ins, *result);
        return true;
    }

    // For multi-dimensional indices, normalize to 1D
    // Step 1: Flatten indices to 1D
    std::size_t total_indices = product_of(ctx.idims());

    // Step 2: Save the original multi-dimensional target shape before normalization
    auto original_target_shape = ctx.target_shape();

    // Step 3: Create 1D context with reshaped indices
    auto ctx_1d = gather_context::with_reshaped_indices(ctx, {total_indices});

    // Step 4: Try optimization with 1D context, passing the original shape
    auto result_1d = try_segment_based_optimization_1d(ctx_1d, builder, original_target_shape);
    if(not result_1d.has_value())
        return false;

    // Step 5: Reshape back to multi-dimensional output shape
    // Final output shape is: pre_lens + idims + post_lens (this is ctx.target_shape())
    auto final_result = builder.reshape(*result_1d, to_int64_vec(original_target_shape));
    m.replace_instruction(ins, final_result);
    return true;
}

} // namespace

struct find_gather
{
    auto matcher() const
    {
        return match::name("gather")(
            match::args(match::any(), match::is_constant().bind("indices")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins          = r.result;
        auto indices_ins  = r.instructions["indices"];
        auto data_ins     = ins->inputs().front();
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

        // Create gather context - pass argument directly
        gather_context ctx(data_ins, axis_index, std::move(indices_arg));

        // Initialize instruction builder
        gather_instruction_builder builder(m, ins);

        // Try segment-based optimization
        try_segment_based_optimization(m, ins, ctx, builder);
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
