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

#include <map>
#include <limits>
#include <numeric>

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

        for(auto& idx : indices_values)
        {
            if(idx < 0)
                idx += static_cast<std::int64_t>(axis_len);
            if(idx < 0 or idx >= static_cast<std::int64_t>(axis_len))
                return;
        }

        const auto idims          = indices_shape.lens();
        const std::size_t in_dims = idims.size();
        const std::size_t total   = indices_values.size();
        std::int64_t base         = indices_values.front();

        const std::vector<std::size_t> pre_lens(dlens.begin(), dlens.begin() + axis_index);
        const std::vector<std::size_t> post_lens(dlens.begin() + axis_index + 1, dlens.end());
        std::vector<std::size_t> rest_lens = pre_lens;
        rest_lens.insert(rest_lens.end(), post_lens.begin(), post_lens.end());

        auto to_int64 = [](const std::vector<std::size_t>& lens) {
            std::vector<int64_t> result;
            result.reserve(lens.size());
            std::transform(lens.begin(), lens.end(), std::back_inserter(result), [](auto len) {
                return static_cast<int64_t>(len);
            });
            return result;
        };

        auto product = [](const std::vector<std::size_t>& lens) {
            return std::accumulate(
                lens.begin(), lens.end(), std::size_t{1}, [](auto acc, auto v) { return acc * v; });
        };

        auto factorize = [](std::size_t value) {
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
        };

        constexpr std::size_t max_factorizations = 256;

        auto enumerate_factorizations = [&](std::size_t value, std::size_t max_results) {
            std::vector<std::vector<std::size_t>> results;
            if(value <= 1)
            {
                results.push_back({value});
                return results;
            }

            std::vector<std::size_t> current;
            const auto dfs =
                [&](auto&& self, std::size_t remaining, std::size_t min_factor) -> void {
                for(std::size_t f = min_factor; f * f <= remaining; ++f)
                {
                    if(remaining % f != 0)
                        continue;
                    if(results.size() >= max_results)
                        return;
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
        };

        std::vector<std::vector<std::size_t>> factor_candidates;
        auto add_candidate = [&](std::vector<std::size_t> factors) {
            if(factors.empty())
                return;
            if(product(factors) != axis_len)
                return;
            if(factors.size() > 8)
                return;
            if(factor_candidates.size() >= max_factorizations)
                return;
            if(std::find(factor_candidates.begin(), factor_candidates.end(), factors) ==
               factor_candidates.end())
            {
                factor_candidates.push_back(std::move(factors));
            }
        };

        for(auto factors : enumerate_factorizations(axis_len, max_factorizations))
        {
            if(factor_candidates.size() >= max_factorizations)
                break;
            add_candidate(std::move(factors));
        }

        if(dlens.size() == 1 and axis_index == 0)
        {
            instruction_ref curr_data = data_ins;
            while(curr_data->name() == "reshape" and curr_data->inputs().size() == 1)
            {
                auto input          = curr_data->inputs().front();
                const auto& in_lens = input->get_shape().lens();
                if(product(in_lens) == axis_len)
                {
                    std::vector<std::size_t> shape_factors;
                    for(auto len : in_lens)
                    {
                        if(len == 1)
                            continue;
                        auto dim_factors = factorize(len);
                        if(dim_factors.empty())
                            dim_factors.push_back(len);
                        shape_factors.insert(
                            shape_factors.end(), dim_factors.begin(), dim_factors.end());
                    }
                    if(not shape_factors.empty())
                    {
                        if(factor_candidates.size() >= max_factorizations)
                            break;
                        add_candidate(std::move(shape_factors));
                    }
                    break;
                }
                curr_data = input;
            }
        }

        auto try_permutation_rewrite = [&]() -> bool {
            if(total != axis_len or axis_len <= 1)
                return false;

            std::vector<std::size_t> sorted_indices;
            sorted_indices.reserve(indices_values.size());
            for(auto v : indices_values)
            {
                if(v < 0)
                    return false;
                sorted_indices.push_back(static_cast<std::size_t>(v));
            }
            std::sort(sorted_indices.begin(), sorted_indices.end());
            for(std::size_t i = 0; i < sorted_indices.size(); ++i)
            {
                if(sorted_indices[i] != i)
                    return false;
            }

            bool is_identity = true;
            for(std::size_t i = 0; i < total; ++i)
            {
                if(static_cast<std::size_t>(indices_values[i]) != i)
                {
                    is_identity = false;
                    break;
                }
            }
            if(is_identity)
                return false;

            if(axis_index == 0 and total == axis_len and axis_len % 2 == 0)
            {
                const std::size_t half = axis_len / 2;
                bool half_shift        = true;
                for(std::size_t i = 0; i < indices_values.size(); ++i)
                {
                    auto expected = (i + half) % axis_len;
                    if(static_cast<std::size_t>(indices_values[i]) != expected)
                    {
                        half_shift = false;
                        break;
                    }
                }
                if(half_shift)
                    return false;
            }

            if(factor_candidates.empty())
                return false;

            std::vector<std::size_t> chosen_factors;
            std::vector<std::size_t> matched_perm;

            auto compute_order = [&](const std::vector<std::size_t>& factor_dims,
                                     const std::vector<std::size_t>& permutation) {
                std::vector<std::size_t> dims_perm;
                dims_perm.reserve(permutation.size());
                for(auto axis : permutation)
                    dims_perm.push_back(factor_dims.at(axis));

                std::vector<std::size_t> coord(permutation.size(), 0);
                std::vector<std::size_t> order;
                order.reserve(axis_len);

                for(std::size_t count = 0; count < axis_len; ++count)
                {
                    std::vector<std::size_t> orig_coord(factor_dims.size(), 0);
                    for(std::size_t i = 0; i < permutation.size(); ++i)
                        orig_coord[permutation[i]] = coord[i];

                    std::size_t idx = 0;
                    for(std::size_t i = 0; i < factor_dims.size(); ++i)
                        idx = idx * factor_dims[i] + orig_coord[i];
                    order.push_back(idx);

                    std::size_t pos = coord.size();
                    while(pos > 0)
                    {
                        --pos;
                        coord[pos]++;
                        if(coord[pos] < dims_perm[pos])
                            break;
                        coord[pos] = 0;
                    }
                }
                return order;
            };

            for(auto factors : factor_candidates)
            {
                if(factors.empty() or factors.size() > 8)
                    continue;

                std::vector<std::size_t> perm(factors.size());
                std::iota(perm.begin(), perm.end(), 0);

                do
                {
                    auto order = compute_order(factors, perm);
                    bool match = true;
                    for(std::size_t i = 0; i < order.size(); ++i)
                    {
                        if(order[i] != static_cast<std::size_t>(indices_values[i]))
                        {
                            match = false;
                            break;
                        }
                    }
                    if(match)
                    {
                        matched_perm   = perm;
                        chosen_factors = std::move(factors);
                        break;
                    }
                } while(std::next_permutation(perm.begin(), perm.end()) and matched_perm.empty());

                if(not matched_perm.empty())
                    break;
            }

            if(matched_perm.empty())
                return false;

            instruction_ref curr = data_ins;

            if(axis_index != 0)
            {
                std::vector<int64_t> perm_axis_front;
                perm_axis_front.reserve(dlens.size());
                perm_axis_front.push_back(static_cast<int64_t>(axis_index));
                for(std::size_t i = 0; i < dlens.size(); ++i)
                {
                    if(i == axis_index)
                        continue;
                    perm_axis_front.push_back(static_cast<int64_t>(i));
                }
                curr = m.insert_instruction(
                    ins, make_op("transpose", {{"permutation", perm_axis_front}}), curr);
            }

            std::vector<int64_t> rest_dims = to_int64(rest_lens);
            std::vector<int64_t> reshape1_dims;
            reshape1_dims.reserve(chosen_factors.size() + rest_dims.size());
            for(auto f : chosen_factors)
                reshape1_dims.push_back(static_cast<int64_t>(f));
            reshape1_dims.insert(reshape1_dims.end(), rest_dims.begin(), rest_dims.end());
            curr = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape1_dims}}), curr);

            if(chosen_factors.size() > 1)
            {
                std::vector<int64_t> perm_extended(chosen_factors.size() + rest_dims.size());
                for(std::size_t i = 0; i < matched_perm.size(); ++i)
                    perm_extended[i] = static_cast<int64_t>(matched_perm[i]);
                for(std::size_t i = 0; i < rest_dims.size(); ++i)
                    perm_extended[matched_perm.size() + i] =
                        static_cast<int64_t>(matched_perm.size() + i);

                bool need_transpose = false;
                for(std::size_t i = 0; i < perm_extended.size(); ++i)
                {
                    if(perm_extended[i] != static_cast<int64_t>(i))
                    {
                        need_transpose = true;
                        break;
                    }
                }
                if(need_transpose)
                {
                    curr = m.insert_instruction(
                        ins, make_op("transpose", {{"permutation", perm_extended}}), curr);
                }
            }

            std::vector<int64_t> reshape2_dims;
            reshape2_dims.reserve(idims.size() + rest_dims.size());
            for(auto dim : idims)
                reshape2_dims.push_back(static_cast<int64_t>(dim));
            reshape2_dims.insert(reshape2_dims.end(), rest_dims.begin(), rest_dims.end());
            curr = m.insert_instruction(ins, make_op("reshape", {{"dims", reshape2_dims}}), curr);

            const std::size_t axis_block_size = idims.size();
            const std::size_t rest_count      = rest_lens.size();
            if(axis_block_size + rest_count > 0)
            {
                std::vector<int64_t> perm_final(axis_block_size + rest_count);
                std::size_t pos = 0;
                for(std::size_t i = 0; i < pre_lens.size(); ++i)
                    perm_final[pos++] = static_cast<int64_t>(axis_block_size + i);
                for(std::size_t i = 0; i < axis_block_size; ++i)
                    perm_final[pos++] = static_cast<int64_t>(i);
                for(std::size_t i = 0; i < post_lens.size(); ++i)
                    perm_final[pos++] = static_cast<int64_t>(axis_block_size + pre_lens.size() + i);

                bool need_transpose = false;
                for(std::size_t i = 0; i < perm_final.size(); ++i)
                {
                    if(perm_final[i] != static_cast<int64_t>(i))
                    {
                        need_transpose = true;
                        break;
                    }
                }
                if(need_transpose)
                {
                    curr = m.insert_instruction(
                        ins, make_op("transpose", {{"permutation", perm_final}}), curr);
                }
            }

            if(curr->get_shape().lens() != ins->get_shape().lens())
            {
                if(curr->get_shape().elements() == ins->get_shape().elements())
                {
                    curr = m.insert_instruction(
                        ins,
                        make_op("reshape", {{"dims", to_int64(ins->get_shape().lens())}}),
                        curr);
                }
                else
                {
                    curr = m.insert_instruction(
                        ins,
                        make_op("multibroadcast",
                                {{"out_lens", to_int64(ins->get_shape().lens())}}),
                        curr);
                }
            }

            m.replace_instruction(ins, curr);
            return true;
        };

        if(try_permutation_rewrite())
            return;

        auto try_stride_slice = [&]() -> bool {
            const std::size_t count = indices_values.size();
            if(count < 2)
                return false;

            if(indices_values.front() != 0)
                return false;

            const std::int64_t stride = indices_values[1] - indices_values[0];
            if(stride <= 1)
                return false;

            for(std::size_t i = 1; i < count; ++i)
            {
                if(indices_values[i] - indices_values[i - 1] != stride)
                    return false;
                if(indices_values[i] != static_cast<std::int64_t>(i) * stride)
                    return false;
            }

            if(axis_len % static_cast<std::size_t>(stride) != 0)
                return false;

            const std::size_t outer = axis_len / static_cast<std::size_t>(stride);
            if(count != outer)
                return false;

            std::vector<int64_t> reshape_dims;
            reshape_dims.reserve(pre_lens.size() + 2 + post_lens.size());
            for(auto len : pre_lens)
                reshape_dims.push_back(static_cast<int64_t>(len));
            reshape_dims.push_back(static_cast<int64_t>(outer));
            reshape_dims.push_back(stride);
            for(auto len : post_lens)
                reshape_dims.push_back(static_cast<int64_t>(len));

            auto reshape =
                m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), data_ins);

            auto slice_axis = static_cast<int64_t>(pre_lens.size() + 1);
            auto slice      = m.insert_instruction(ins,
                                              make_op("slice",
                                                           {{"axes", std::vector<int64_t>{slice_axis}},
                                                            {"starts", std::vector<int64_t>{0}},
                                                            {"ends", std::vector<int64_t>{1}}}),
                                              reshape);

            auto result = m.insert_instruction(
                ins, make_op("reshape", {{"dims", to_int64(ins->get_shape().lens())}}), slice);

            m.replace_instruction(ins, result);
            return true;
        };

        if(try_stride_slice())
            return;

        auto try_rectangular_rewrite = [&]() -> bool {
            if(factor_candidates.empty())
                return false;

            if(axis_index == 0 and total == axis_len and axis_len % 2 == 0)
            {
                const std::size_t half = axis_len / 2;
                bool half_shift        = true;
                for(std::size_t i = 0; i < indices_values.size(); ++i)
                {
                    auto expected = (i + half) % axis_len;
                    if(static_cast<std::size_t>(indices_values[i]) != expected)
                    {
                        half_shift = false;
                        break;
                    }
                }
                if(half_shift)
                    return false;
            }

            const auto invalid_index_value = std::numeric_limits<std::size_t>::max();
            std::vector<int64_t> rest_dims = to_int64(rest_lens);

            for(auto factors : factor_candidates)
            {
                if(factors.empty() or product(factors) != axis_len)
                    continue;

                std::vector<std::size_t> perm(factors.size());
                std::iota(perm.begin(), perm.end(), 0);

                do
                {
                    std::vector<std::size_t> dims_perm(perm.size());
                    for(std::size_t i = 0; i < perm.size(); ++i)
                        dims_perm[i] = factors[perm[i]];

                    std::vector<std::vector<std::size_t>> coords(
                        total, std::vector<std::size_t>(perm.size()));
                    bool consistent = true;
                    for(std::size_t idx = 0; idx < total and consistent; ++idx)
                    {
                        auto value = static_cast<std::size_t>(indices_values[idx]);
                        std::vector<std::size_t> coord(factors.size());
                        auto remainder = value;
                        for(std::size_t j = factors.size(); j > 0; --j)
                        {
                            auto dim_index   = j - 1;
                            auto dim_size    = factors[dim_index];
                            coord[dim_index] = remainder % dim_size;
                            remainder /= dim_size;
                        }
                        if(remainder != 0)
                        {
                            consistent = false;
                            break;
                        }
                        for(std::size_t j = 0; j < perm.size(); ++j)
                            coords[idx][j] = coord[perm[j]];
                    }
                    if(not consistent)
                        continue;

                    std::vector<std::size_t> min_coord(dims_perm.size(),
                                                       std::numeric_limits<std::size_t>::max());
                    std::vector<std::size_t> max_coord(dims_perm.size(), 0);
                    for(auto& c : coords)
                    {
                        for(std::size_t j = 0; j < c.size(); ++j)
                        {
                            min_coord[j] = std::min(min_coord[j], c[j]);
                            max_coord[j] = std::max(max_coord[j], c[j]);
                        }
                    }

                    std::vector<std::size_t> len(dims_perm.size(), 0);
                    std::size_t block_total = 1;
                    for(std::size_t j = 0; j < len.size(); ++j)
                    {
                        if(min_coord[j] > max_coord[j])
                        {
                            consistent = false;
                            break;
                        }
                        len[j] = max_coord[j] - min_coord[j] + 1;
                        if(len[j] > dims_perm[j])
                        {
                            consistent = false;
                            break;
                        }
                        block_total *= len[j];
                    }
                    if(not consistent or block_total != total)
                        continue;

                    std::unordered_set<std::size_t> seen;
                    seen.reserve(total * 2);
                    for(auto& c : coords)
                    {
                        std::size_t block_idx = 0;
                        for(std::size_t j = 0; j < len.size(); ++j)
                        {
                            auto offset = c[j] - min_coord[j];
                            if(offset >= len[j])
                            {
                                consistent = false;
                                break;
                            }
                            block_idx = block_idx * len[j] + offset;
                        }
                        if(not consistent)
                            break;
                        seen.insert(block_idx);
                    }
                    if(not consistent or seen.size() != total)
                        continue;

                    std::vector<int> axis_to_index(len.size(), -1);
                    std::vector<bool> used_index(in_dims, false);
                    for(std::size_t axis_dim = 0; axis_dim < len.size() and consistent; ++axis_dim)
                    {
                        int chosen_index = -1;
                        for(std::size_t index_dim = 0; index_dim < in_dims; ++index_dim)
                        {
                            if(used_index[index_dim])
                                continue;
                            if(idims[index_dim] != len[axis_dim])
                                continue;
                            std::vector<std::size_t> value_per_coord(idims[index_dim],
                                                                     invalid_index_value);
                            bool axis_matches = true;
                            for(std::size_t idx = 0; idx < total; ++idx)
                            {
                                auto coord_index = indices_shape.multi(idx);
                                auto axis_value  = coords[idx][axis_dim];
                                auto coord_value = coord_index[index_dim];
                                auto& slot       = value_per_coord[coord_value];
                                if(slot == invalid_index_value)
                                    slot = axis_value;
                                else if(slot != axis_value)
                                {
                                    axis_matches = false;
                                    break;
                                }
                            }
                            if(axis_matches)
                            {
                                chosen_index            = static_cast<int>(index_dim);
                                axis_to_index[axis_dim] = chosen_index;
                                used_index[index_dim]   = true;
                                break;
                            }
                        }
                        if(chosen_index == -1)
                        {
                            consistent = false;
                            break;
                        }
                    }
                    if(not consistent)
                        continue;

                    instruction_ref curr = data_ins;

                    if(axis_index != 0)
                    {
                        std::vector<int64_t> perm_axis_front;
                        perm_axis_front.reserve(dlens.size());
                        perm_axis_front.push_back(static_cast<int64_t>(axis_index));
                        for(std::size_t i = 0; i < dlens.size(); ++i)
                        {
                            if(i == axis_index)
                                continue;
                            perm_axis_front.push_back(static_cast<int64_t>(i));
                        }
                        curr = m.insert_instruction(
                            ins, make_op("transpose", {{"permutation", perm_axis_front}}), curr);
                    }

                    std::vector<int64_t> reshape_axis_dims;
                    reshape_axis_dims.reserve(factors.size() + rest_dims.size());
                    for(auto f : factors)
                        reshape_axis_dims.push_back(static_cast<int64_t>(f));
                    reshape_axis_dims.insert(
                        reshape_axis_dims.end(), rest_dims.begin(), rest_dims.end());
                    curr = m.insert_instruction(
                        ins, make_op("reshape", {{"dims", reshape_axis_dims}}), curr);

                    if(factors.size() > 1)
                    {
                        std::vector<int64_t> perm_extended(factors.size() + rest_dims.size());
                        for(std::size_t i = 0; i < perm.size(); ++i)
                            perm_extended[i] = static_cast<int64_t>(perm[i]);
                        for(std::size_t i = 0; i < rest_dims.size(); ++i)
                            perm_extended[perm.size() + i] = static_cast<int64_t>(perm.size() + i);

                        bool need_transpose = false;
                        for(std::size_t i = 0; i < perm_extended.size(); ++i)
                        {
                            if(perm_extended[i] != static_cast<int64_t>(i))
                            {
                                need_transpose = true;
                                break;
                            }
                        }
                        if(need_transpose)
                        {
                            curr = m.insert_instruction(
                                ins, make_op("transpose", {{"permutation", perm_extended}}), curr);
                        }
                    }

                    std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> slice_desc;
                    for(std::size_t j = 0; j < min_coord.size(); ++j)
                    {
                        auto start = static_cast<int64_t>(min_coord[j]);
                        auto end   = static_cast<int64_t>(min_coord[j] + len[j]);
                        if(start != 0 or end != static_cast<int64_t>(dims_perm[j]))
                            slice_desc.push_back({static_cast<int64_t>(j), {start, end}});
                    }
                    if(not slice_desc.empty())
                    {
                        std::sort(slice_desc.begin(),
                                  slice_desc.end(),
                                  [](const auto& a, const auto& b) { return a.first < b.first; });
                        std::vector<int64_t> axes;
                        std::vector<int64_t> starts;
                        std::vector<int64_t> ends;
                        axes.reserve(slice_desc.size());
                        starts.reserve(slice_desc.size());
                        ends.reserve(slice_desc.size());
                        for(auto& s : slice_desc)
                        {
                            axes.push_back(s.first);
                            starts.push_back(s.second.first);
                            ends.push_back(s.second.second);
                        }
                        curr = m.insert_instruction(
                            ins,
                            make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                            curr);
                    }

                    if(axis_to_index.size() > 1)
                    {
                        std::vector<std::size_t> dims_for_index(axis_to_index.size());
                        for(std::size_t j = 0; j < axis_to_index.size(); ++j)
                            dims_for_index[static_cast<std::size_t>(axis_to_index[j])] = j;

                        bool need_reorder = false;
                        for(std::size_t k = 0; k < dims_for_index.size(); ++k)
                        {
                            if(dims_for_index[k] != k)
                            {
                                need_reorder = true;
                                break;
                            }
                        }
                        if(need_reorder)
                        {
                            std::vector<int64_t> perm_align(axis_to_index.size() +
                                                            rest_dims.size());
                            for(std::size_t k = 0; k < dims_for_index.size(); ++k)
                                perm_align[k] = static_cast<int64_t>(dims_for_index[k]);
                            for(std::size_t i = 0; i < rest_dims.size(); ++i)
                                perm_align[axis_to_index.size() + i] =
                                    static_cast<int64_t>(axis_to_index.size() + i);
                            curr = m.insert_instruction(
                                ins, make_op("transpose", {{"permutation", perm_align}}), curr);
                        }
                    }

                    const std::size_t axis_block_size = in_dims;
                    const std::size_t rest_count      = rest_lens.size();
                    if(axis_block_size + rest_count > 0)
                    {
                        std::vector<int64_t> perm_final(axis_block_size + rest_count);
                        std::size_t pos = 0;
                        for(std::size_t i = 0; i < pre_lens.size(); ++i)
                            perm_final[pos++] = static_cast<int64_t>(axis_block_size + i);
                        for(std::size_t i = 0; i < axis_block_size; ++i)
                            perm_final[pos++] = static_cast<int64_t>(i);
                        for(std::size_t i = 0; i < post_lens.size(); ++i)
                            perm_final[pos++] =
                                static_cast<int64_t>(axis_block_size + pre_lens.size() + i);

                        bool need_transpose = false;
                        for(std::size_t i = 0; i < perm_final.size(); ++i)
                        {
                            if(perm_final[i] != static_cast<int64_t>(i))
                            {
                                need_transpose = true;
                                break;
                            }
                        }
                        if(need_transpose)
                        {
                            curr = m.insert_instruction(
                                ins, make_op("transpose", {{"permutation", perm_final}}), curr);
                        }
                    }

                    if(curr->get_shape().lens() != ins->get_shape().lens())
                    {
                        if(curr->get_shape().elements() == ins->get_shape().elements())
                        {
                            curr = m.insert_instruction(
                                ins,
                                make_op("reshape", {{"dims", to_int64(ins->get_shape().lens())}}),
                                curr);
                        }
                        else
                        {
                            curr = m.insert_instruction(
                                ins,
                                make_op("multibroadcast",
                                        {{"out_lens", to_int64(ins->get_shape().lens())}}),
                                curr);
                        }
                    }

                    m.replace_instruction(ins, curr);
                    return true;
                } while(std::next_permutation(perm.begin(), perm.end()));
            }

            return false;
        };

        if(try_rectangular_rewrite())
            return;

        auto try_tile_rewrite = [&]() -> bool {
            std::vector<std::size_t> repeat_sizes(in_dims, 1);
            std::vector<std::size_t> tile_sizes(in_dims, 1);
            auto is_repeated_axis = [&](std::size_t axis, std::size_t repeat) {
                if(repeat <= 1)
                    return false;
                auto axis_len_dim = idims[axis];
                if(axis_len_dim % repeat != 0)
                    return false;
                for(std::size_t idx = 0; idx < total; ++idx)
                {
                    auto coord    = indices_shape.multi(idx);
                    auto axis_val = coord[axis];
                    auto group    = axis_val / repeat;
                    coord[axis]   = group * repeat;
                    auto base_idx = indices_shape.index(coord);
                    if(indices_values[idx] != indices_values[base_idx])
                        return false;
                }
                return true;
            };

            for(std::size_t dim = 0; dim < in_dims; ++dim)
            {
                auto axis_len_dim  = idims[dim];
                std::size_t repeat = 1;
                for(std::size_t candidate = 2; candidate <= axis_len_dim; ++candidate)
                {
                    if(axis_len_dim % candidate != 0)
                        continue;
                    if(is_repeated_axis(dim, candidate))
                    {
                        repeat = candidate;
                        break;
                    }
                }
                repeat_sizes[dim] = repeat;
                tile_sizes[dim]   = (repeat > 0) ? axis_len_dim / repeat : 0;
                if(tile_sizes[dim] == 0)
                    return false;
            }

            std::vector<std::size_t> tile_axes;
            std::size_t tile_product = 1;
            for(std::size_t dim = 0; dim < in_dims; ++dim)
            {
                if(tile_sizes[dim] > 1)
                {
                    tile_axes.push_back(dim);
                    tile_product *= tile_sizes[dim];
                }
            }

            const bool broadcast_needed = std::any_of(
                repeat_sizes.begin(), repeat_sizes.end(), [](std::size_t r) { return r > 1; });

            std::vector<std::int64_t> strides(in_dims, 0);
            std::size_t weight = 1;
            for(auto it = tile_axes.rbegin(); it != tile_axes.rend(); ++it)
            {
                strides[*it] = static_cast<std::int64_t>(weight);
                weight *= tile_sizes[*it];
            }

            for(std::size_t idx = 0; idx < total; ++idx)
            {
                auto coord            = indices_shape.multi(idx);
                std::int64_t expected = 0;
                for(auto axis : tile_axes)
                {
                    auto tile_index = coord[axis] / repeat_sizes[axis];
                    expected += strides[axis] * static_cast<std::int64_t>(tile_index);
                }
                if(indices_values[idx] - base != expected)
                    return false;
            }

            std::int64_t max_index = base;
            for(auto axis : tile_axes)
            {
                max_index += strides[axis] * static_cast<std::int64_t>(tile_sizes[axis] - 1);
            }

            if(base < 0 or max_index < base)
                return false;
            if(max_index >= static_cast<std::int64_t>(axis_len))
                return false;

            auto slice_len = max_index - base + 1;
            if(slice_len <= 0)
                return false;

            const auto slice_len_size = static_cast<std::size_t>(slice_len);
            if(slice_len_size == 0)
                return false;

            const bool has_tiled_repeat =
                std::any_of(tile_axes.begin(), tile_axes.end(), [&](std::size_t dim) {
                    return repeat_sizes[dim] > 1;
                });
            if(slice_len_size != axis_len && has_tiled_repeat)
                return false;

            if(tile_axes.empty())
            {
                if(slice_len_size != 1)
                    return false;
            }
            else if(tile_product != slice_len_size)
            {
                return false;
            }

            std::vector<std::size_t> vary_dims = tile_axes;

            std::size_t prod_vary = 1;
            for(auto dim : vary_dims)
                prod_vary *= tile_sizes[dim];
            if(static_cast<std::size_t>(slice_len) != prod_vary and not vary_dims.empty())
                return false;

            std::vector<std::size_t> sorted_vary = vary_dims;
            std::sort(sorted_vary.begin(), sorted_vary.end(), [&](std::size_t a, std::size_t b) {
                return strides[a] < strides[b];
            });

            std::int64_t expected_stride = 1;
            for(auto dim : sorted_vary)
            {
                if(strides[dim] != expected_stride)
                    return false;
                expected_stride *= static_cast<std::int64_t>(tile_sizes[dim]);
            }
            if(not sorted_vary.empty() and expected_stride != slice_len)
                return false;

            std::vector<std::size_t> ordered_vary_desc = sorted_vary;
            std::reverse(ordered_vary_desc.begin(), ordered_vary_desc.end());
            std::vector<std::size_t> target_vary_order = vary_dims;

            const auto& output_lens = ins->get_shape().lens();

            instruction_ref curr = data_ins;

            if(axis_index != 0)
            {
                std::vector<int64_t> perm_axis_front;
                perm_axis_front.reserve(dlens.size());
                perm_axis_front.push_back(static_cast<int64_t>(axis_index));
                for(std::size_t i = 0; i < dlens.size(); ++i)
                {
                    if(i == axis_index)
                        continue;
                    perm_axis_front.push_back(static_cast<int64_t>(i));
                }
                curr = m.insert_instruction(
                    ins, make_op("transpose", {{"permutation", perm_axis_front}}), curr);
            }

            if(base != 0 or static_cast<std::size_t>(slice_len) != axis_len)
            {
                std::vector<int64_t> axes{0};
                std::vector<int64_t> starts{base};
                std::vector<int64_t> ends{base + slice_len};
                curr = m.insert_instruction(
                    ins,
                    make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                    curr);
            }

            std::vector<int64_t> rest_dims;
            rest_dims.reserve(rest_lens.size());
            std::transform(rest_lens.begin(),
                           rest_lens.end(),
                           std::back_inserter(rest_dims),
                           [](auto len) { return static_cast<int64_t>(len); });

            if(not ordered_vary_desc.empty())
            {
                std::vector<int64_t> reshape1_dims;
                reshape1_dims.reserve(ordered_vary_desc.size() + rest_dims.size());
                for(auto dim : ordered_vary_desc)
                    reshape1_dims.push_back(static_cast<int64_t>(tile_sizes[dim]));
                reshape1_dims.insert(reshape1_dims.end(), rest_dims.begin(), rest_dims.end());
                curr =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", reshape1_dims}}), curr);

                if(ordered_vary_desc != target_vary_order)
                {
                    const std::size_t axis_count = ordered_vary_desc.size();
                    std::vector<int64_t> perm(axis_count + rest_dims.size());
                    for(std::size_t i = 0; i < target_vary_order.size(); ++i)
                    {
                        auto it = std::find(ordered_vary_desc.begin(),
                                            ordered_vary_desc.end(),
                                            target_vary_order[i]);
                        if(it == ordered_vary_desc.end())
                            return false;
                        perm[i] = std::distance(ordered_vary_desc.begin(), it);
                    }
                    for(std::size_t i = 0; i < rest_dims.size(); ++i)
                        perm[target_vary_order.size() + i] = static_cast<int64_t>(axis_count + i);

                    curr = m.insert_instruction(
                        ins, make_op("transpose", {{"permutation", perm}}), curr);
                    ordered_vary_desc = target_vary_order;
                }
            }

            if(in_dims > 0)
            {
                std::vector<int64_t> reshape2_dims;
                reshape2_dims.reserve(in_dims + rest_dims.size());
                for(std::size_t dim = 0; dim < in_dims; ++dim)
                {
                    if(tile_sizes[dim] > 1)
                        reshape2_dims.push_back(static_cast<int64_t>(tile_sizes[dim]));
                    else
                        reshape2_dims.push_back(1);

                    if(repeat_sizes[dim] > 1)
                        reshape2_dims.push_back(1);
                }
                reshape2_dims.insert(reshape2_dims.end(), rest_dims.begin(), rest_dims.end());
                if(reshape2_dims.empty())
                    reshape2_dims.push_back(1);
                curr =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", reshape2_dims}}), curr);
                if(broadcast_needed)
                {
                    std::vector<int64_t> broadcast_dims;
                    broadcast_dims.reserve(in_dims + rest_dims.size());
                    for(std::size_t dim = 0; dim < in_dims; ++dim)
                    {
                        auto tile_val =
                            (tile_sizes[dim] > 1) ? static_cast<int64_t>(tile_sizes[dim]) : 1;
                        broadcast_dims.push_back(tile_val);
                        if(repeat_sizes[dim] > 1)
                            broadcast_dims.push_back(static_cast<int64_t>(repeat_sizes[dim]));
                    }
                    broadcast_dims.insert(broadcast_dims.end(), rest_dims.begin(), rest_dims.end());
                    curr = m.insert_instruction(
                        ins, make_op("multibroadcast", {{"out_lens", broadcast_dims}}), curr);
                }

                std::vector<int64_t> combine_dims;
                combine_dims.reserve(in_dims + rest_dims.size());
                for(std::size_t dim = 0; dim < in_dims; ++dim)
                {
                    auto tile_val   = (tile_sizes[dim] > 1) ? tile_sizes[dim] : std::size_t{1};
                    auto repeat_val = repeat_sizes[dim];
                    combine_dims.push_back(static_cast<int64_t>(tile_val * repeat_val));
                }
                combine_dims.insert(combine_dims.end(), rest_dims.begin(), rest_dims.end());
                if(combine_dims.empty())
                    combine_dims.push_back(1);
                curr =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", combine_dims}}), curr);
            }

            const std::size_t axis_block_size = in_dims;
            const std::size_t pre_count       = pre_lens.size();
            const std::size_t post_count      = post_lens.size();
            const std::size_t rest_count      = rest_dims.size();

            if(axis_block_size + rest_count > 0)
            {
                std::vector<int64_t> perm_final(axis_block_size + rest_count);
                std::size_t pos = 0;
                for(std::size_t i = 0; i < pre_count; ++i)
                    perm_final[pos++] = static_cast<int64_t>(axis_block_size + i);
                for(std::size_t i = 0; i < axis_block_size; ++i)
                    perm_final[pos++] = static_cast<int64_t>(i);
                for(std::size_t i = 0; i < post_count; ++i)
                    perm_final[pos++] = static_cast<int64_t>(axis_block_size + pre_count + i);

                bool need_transpose = false;
                for(std::size_t i = 0; i < perm_final.size(); ++i)
                {
                    if(perm_final[i] != static_cast<int64_t>(i))
                    {
                        need_transpose = true;
                        break;
                    }
                }
                if(need_transpose)
                {
                    curr = m.insert_instruction(
                        ins, make_op("transpose", {{"permutation", perm_final}}), curr);
                }
            }

            if(curr->get_shape().lens() != output_lens)
            {
                if(curr->get_shape().elements() == ins->get_shape().elements())
                {
                    curr = m.insert_instruction(
                        ins, make_op("reshape", {{"dims", to_int64(output_lens)}}), curr);
                }
                else
                {
                    curr = m.insert_instruction(
                        ins, make_op("multibroadcast", {{"out_lens", output_lens}}), curr);
                }
            }

            m.replace_instruction(ins, curr);
            return true;
        };

        if(try_tile_rewrite())
            return;

        auto try_half_split_concat = [&]() -> bool {
            if(axis_index != 0)
                return false;

            if(total != axis_len)
                return false;

            if(axis_len <= 1 or axis_len % 2 != 0)
                return false;

            std::vector<std::size_t> sorted(indices_values.size());
            std::transform(indices_values.begin(),
                           indices_values.end(),
                           sorted.begin(),
                           [](auto v) { return static_cast<std::size_t>(v); });
            std::sort(sorted.begin(), sorted.end());
            for(std::size_t i = 0; i < sorted.size(); ++i)
            {
                if(sorted[i] != i)
                    return false;
            }

            const std::size_t half = axis_len / 2;
            for(std::size_t i = 0; i < indices_values.size(); ++i)
            {
                auto expected = (i + half) % axis_len;
                if(static_cast<std::size_t>(indices_values[i]) != expected)
                    return false;
            }

            std::vector<int64_t> axes{0};
            const auto half_i64     = static_cast<int64_t>(half);
            const auto axis_len_i64 = static_cast<int64_t>(axis_len);

            auto tail = m.insert_instruction(
                ins,
                make_op("slice",
                        {{"axes", axes}, {"starts", {half_i64}}, {"ends", {axis_len_i64}}}),
                data_ins);
            auto head = m.insert_instruction(
                ins,
                make_op("slice", {{"axes", axes}, {"starts", {0}}, {"ends", {half_i64}}}),
                data_ins);

            auto concat =
                m.insert_instruction(ins, make_op("concat", {{"axis", int64_t{0}}}), tail, head);

            std::vector<int64_t> reshape_dims = to_int64(idims);
            auto rest_dims                    = to_int64(rest_lens);
            reshape_dims.insert(reshape_dims.end(), rest_dims.begin(), rest_dims.end());

            instruction_ref curr = concat;
            if(curr->get_shape().lens() != ins->get_shape().lens())
            {
                if(reshape_dims.empty())
                    reshape_dims.push_back(1);
                curr =
                    m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), curr);
            }

            m.replace_instruction(ins, curr);
            return true;
        };

        if(try_half_split_concat())
            return;
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
