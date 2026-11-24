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
#include <migraphx/rewrite_nd_convolution.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// One spatial dim of a convolution
struct conv_dim
{
    std::size_t in_len   = 1;
    std::size_t out_len  = 1;
    std::size_t taps     = 1;
    std::size_t padding  = 0;
    std::size_t stride   = 1;
    std::size_t dilation = 1;

    // The tap output `out` reads input `in` through (in = out * stride + tap * dilation -
    // padding), or `taps` when it reads none
    std::size_t tap(std::size_t out, std::size_t in) const
    {
        auto pos    = in + padding;
        auto origin = out * stride;
        if(pos < origin or (pos - origin) % dilation != 0)
            return taps;
        return std::min((pos - origin) / dilation, taps);
    }

    // Tap for each (output, input) pair, output-major
    std::vector<int64_t> fold_taps() const
    {
        shape s{shape::int64_type, {out_len, in_len}};
        std::vector<int64_t> result(s.elements());
        auto elements = range(s.elements());
        std::transform(elements.begin(), elements.end(), result.begin(), [&](auto i) {
            auto idx = s.multi(i);
            return tap(idx[0], idx[1]);
        });
        return result;
    }
};

struct nd_convolution
{
    instruction_ref ins;
    value attributes;
    std::vector<conv_dim> dims;

    static nd_convolution create(instruction_ref ins)
    {
        nd_convolution result{ins, ins->get_operator().to_value(), {}};
        const auto& x_lens   = ins->inputs().front()->get_shape().lens();
        const auto& w_lens   = ins->inputs().back()->get_shape().lens();
        const auto& out_lens = ins->get_shape().lens();
        auto padding         = result.attributes.at("padding").to_vector<std::size_t>();
        auto stride          = result.attributes.at("stride").to_vector<std::size_t>();
        auto dilation        = result.attributes.at("dilation").to_vector<std::size_t>();
        auto spatial         = range(x_lens.size() - 2);
        std::transform(
            spatial.begin(), spatial.end(), std::back_inserter(result.dims), [&](auto i) {
                return conv_dim{x_lens[i + 2],
                                out_lens[i + 2],
                                w_lens[i + 2],
                                padding[i],
                                stride[i],
                                dilation[i]};
            });
        return result;
    }

    instruction_ref input() const { return ins->inputs().front(); }
    instruction_ref weights() const { return ins->inputs().back(); }

    // Spatial dims to fold into the channels, ascending: those where one input element
    // expands to several outputs, then the smallest of the others while more than
    // max_spatial_dims are left
    std::vector<std::size_t> fold_dims(std::size_t max_spatial_dims) const
    {
        std::vector<std::size_t> result(dims.size());
        std::iota(result.begin(), result.end(), 0);
        auto others = std::stable_partition(result.begin(), result.end(), [&](auto i) {
            return dims[i].in_len == 1 and dims[i].out_len > 1;
        });
        std::stable_sort(others, result.end(), [&](auto i, auto j) {
            return dims[i].in_len * dims[i].out_len < dims[j].in_len * dims[j].out_len;
        });
        std::size_t expanding = std::distance(result.begin(), others);
        std::size_t extra     = dims.size() > max_spatial_dims ? dims.size() - max_spatial_dims : 0;
        result.resize(std::max(expanding, extra));
        std::sort(result.begin(), result.end());
        return result;
    }
};

// Whether the channels come after the folded dims in the memory order of the tensor
bool channels_inner(const shape& s, const std::vector<std::size_t>& folded)
{
    auto perm    = find_permutation(s);
    auto channel = std::find(perm.begin(), perm.end(), 1);
    return std::all_of(folded.begin(), folded.end(), [&](auto d) {
        return std::find(perm.begin(), channel, d + 2) != channel;
    });
}

// Dims of an (N, C, spatial...) tensor with the folded spatial dims moved next to the
// channels: after them, or before them when the channels are the inner dim
std::vector<int64_t>
fold_permutation(std::size_t ndim, const std::vector<std::size_t>& folded, bool inner)
{
    std::vector<int64_t> result(ndim);
    std::iota(result.begin(), result.end(), 0);
    std::stable_partition(
        result.begin() + 2, result.end(), [&](auto d) { return contains(folded, d - 2); });
    if(inner)
        std::rotate(result.begin() + 1, result.begin() + 2, result.begin() + 2 + folded.size());
    return result;
}

// The values of the spatial dims that are not folded
template <class Iterator>
std::vector<std::size_t>
remove_folded(Iterator start, const nd_convolution& conv, const std::vector<std::size_t>& folded)
{
    std::vector<std::size_t> result;
    auto spatial = range(conv.dims.size());
    transform_if(
        spatial.begin(),
        spatial.end(),
        std::back_inserter(result),
        [&](auto i) { return not contains(folded, i); },
        [&](auto i) { return start[i]; });
    return result;
}

// Fold spatial dim `axis` of the (K, C, k...) weights into its channels: (K * out_len,
// C * in_len, ...) with the tap that connects each output to each input. The channels are the
// outer or the inner dim of each product.
instruction_ref fold_weights(module& m,
                             instruction_ref pos,
                             instruction_ref w,
                             std::size_t axis,
                             const conv_dim& d,
                             bool inner)
{
    auto taps = d.fold_taps();
    // Pairs that no tap connects read a tap of zeros added after the filter
    if(contains(taps, d.taps))
    {
        std::vector<int64_t> pads(2 * w->get_shape().ndim(), 0);
        pads[w->get_shape().ndim() + axis] = 1;
        w = m.insert_instruction(pos, make_op("pad", {{"pads", pads}}), w);
    }
    auto indices = m.add_literal(literal{shape{shape::int64_type, {taps.size()}}, taps});
    w            = m.insert_instruction(pos, make_op("gather", {{"axis", axis}}), w, indices);

    auto lens = w->get_shape().lens();
    std::vector<int64_t> split(lens.begin(), lens.end());
    split[axis] = d.out_len;
    split.insert(split.begin() + axis + 1, d.in_len);
    // (K, C, ..., out, in, ...) -> (K, out, C, in, ...) or (out, K, in, C, ...)
    std::vector<int64_t> perm(split.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::rotate(perm.begin() + 1, perm.begin() + axis, perm.begin() + axis + 1);
    std::rotate(perm.begin() + 3, perm.begin() + axis + 1, perm.begin() + axis + 2);
    if(inner)
    {
        std::swap(perm[0], perm[1]);
        std::swap(perm[2], perm[3]);
    }
    std::vector<int64_t> merged(lens.begin(), lens.end());
    merged.erase(merged.begin() + axis);
    merged[0] *= d.out_len;
    merged[1] *= d.in_len;

    w = m.insert_instruction(pos, make_op("reshape", {{"dims", split}}), w);
    w = m.insert_instruction(pos, make_op("transpose", {{"permutation", perm}}), w);
    return m.insert_instruction(pos, make_op("reshape", {{"dims", merged}}), w);
}

// Memory order of a tensor's dims without the folded spatial dims
std::vector<int64_t> folded_layout(const shape& s, const std::vector<std::size_t>& folded)
{
    auto perm = find_permutation(s);
    std::vector<int64_t> result;
    transform_if(
        perm.begin(),
        perm.end(),
        std::back_inserter(result),
        [&](auto d) { return d < 2 or not contains(folded, d - 2); },
        [&](auto d) {
            return d -
                   std::count_if(folded.begin(), folded.end(), [&](auto f) { return f + 2 < d; });
        });
    return result;
}

struct find_nd_convolution
{
    std::size_t max_spatial_dims = 3;

    auto matcher() const { return match::name("convolution"); }

    // The bias a single add applies to `ins` when it only varies along the channels and the
    // folded dims, so it can be added to the folded convolution
    static bool
    is_folded_bias(instruction_ref add, instruction_ref ins, const std::vector<std::size_t>& folded)
    {
        if(add->name() != "add" or add->inputs().size() != 2)
            return false;
        auto other = add->inputs().front() == ins ? add->inputs().back() : add->inputs().front();
        if(other == ins or other->name() != "multibroadcast" or other->inputs().size() != 1)
            return false;
        const auto& lens = other->inputs().front()->get_shape().lens();
        auto ndim        = ins->get_shape().ndim();
        if(lens.size() > ndim)
            return false;
        auto offset = ndim - lens.size();
        auto bdims  = range(lens.size());
        return std::all_of(bdims.begin(), bdims.end(), [&](auto i) {
            auto d = i + offset;
            return lens[i] == 1 or d == 1 or (d >= 2 and contains(folded, d - 2));
        });
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        if(ins->get_shape().dynamic() or ins->inputs().back()->get_shape().dynamic())
            return;
        auto conv = nd_convolution::create(ins);
        if(conv.attributes.at("group").to<std::size_t>() != 1 or
           conv.attributes.at("padding_mode").to<int>() != 0)
            return;
        auto folded = conv.fold_dims(max_spatial_dims);
        if(folded.empty())
            return;

        // The folded convolution keeps the channels as the inner dim of its output when the
        // convolution has them after the folded dims, so that a reshape of the result gets the
        // layout it gets from the convolution
        const bool inner     = channels_inner(ins->get_shape(), folded);
        const auto& out_lens = ins->get_shape().lens();
        auto perm            = fold_permutation(out_lens.size(), folded, inner);
        auto fold            = range(folded.size());

        auto x = conv.input();
        if(std::all_of(
               folded.begin(), folded.end(), [&](auto i) { return conv.dims[i].in_len == 1; }))
        {
            std::vector<int64_t> axes(folded.size());
            std::transform(
                folded.begin(), folded.end(), axes.begin(), [](auto i) { return i + 2; });
            x = m.insert_instruction(ins, make_op("squeeze", {{"axes", axes}}), x);
        }
        else
        {
            auto x_lens = reorder_dims(x->get_shape().lens(), perm);
            std::vector<int64_t> x_dims(x_lens.begin() + folded.size(), x_lens.end());
            x_dims[0] = x_lens[0];
            x_dims[1] = std::accumulate(x_lens.begin() + 1,
                                        x_lens.begin() + 2 + folded.size(),
                                        std::size_t{1},
                                        std::multiplies<>{});
            if(not std::is_sorted(perm.begin(), perm.end()))
                x = m.insert_instruction(ins, make_op("transpose", {{"permutation", perm}}), x);
            x = m.insert_instruction(ins, make_op("reshape", {{"dims", x_dims}}), x);
        }

        // Each fold removes a dim, which shifts the later ones. With the channels inner a fold
        // puts its dim outside the ones folded before it, so the last dim is folded first to
        // keep the folded dims in order.
        auto w = conv.weights();
        if(inner)
            w = std::accumulate(folded.rbegin(), folded.rend(), w, [&](auto w_ins, auto d) {
                return fold_weights(m, ins, w_ins, d + 2, conv.dims[d], true);
            });
        else
            w = std::accumulate(fold.begin(), fold.end(), w, [&](auto w_ins, auto i) {
                return fold_weights(m, ins, w_ins, folded[i] + 2 - i, conv.dims[folded[i]], false);
            });
        if(not conv.weights()->get_shape().standard())
            w = m.insert_instruction(
                ins,
                make_op("layout",
                        {{"permutation", folded_layout(conv.weights()->get_shape(), folded)}}),
                w);

        auto padding  = conv.attributes.at("padding").to_vector<std::size_t>();
        auto nspatial = conv.dims.size();
        if(padding.size() == nspatial)
            padding.insert(padding.end(), padding.begin(), padding.begin() + nspatial);
        auto new_padding = remove_folded(padding.begin(), conv, folded);
        auto end_padding = remove_folded(padding.begin() + nspatial, conv, folded);
        new_padding.insert(new_padding.end(), end_padding.begin(), end_padding.end());
        auto v       = conv.attributes;
        v["padding"] = new_padding;
        v["stride"]  = remove_folded(v.at("stride").to_vector<std::size_t>().begin(), conv, folded);
        v["dilation"] =
            remove_folded(v.at("dilation").to_vector<std::size_t>().begin(), conv, folded);

        // The instruction to replace: the convolution, or the last of the contiguous and the
        // bias add that follow it
        auto last   = ins;
        auto result = m.insert_instruction(ins, make_op("convolution", v), x, w);
        if(last->outputs().size() == 1 and last->outputs().front()->name() == "contiguous")
        {
            last   = last->outputs().front();
            result = m.insert_instruction(last, make_op("contiguous"), result);
        }
        // Lens of the output with the folded dims next to the channels
        auto split_lens = reorder_dims(out_lens, perm);
        if(last->outputs().size() == 1 and is_folded_bias(last->outputs().front(), last, folded))
        {
            auto add = last->outputs().front();
            auto bcast =
                add->inputs().front() == last ? add->inputs().back() : add->inputs().front();
            // One value per output channel of the folded convolution
            auto out_dims = range(out_lens.size());
            std::vector<std::size_t> bias_lens(out_lens.size());
            std::transform(
                out_dims.begin(), out_dims.end(), bias_lens.begin(), [&](auto d) -> std::size_t {
                    if(d == 1 or (d >= 2 and contains(folded, d - 2)))
                        return out_lens[d];
                    return 1;
                });
            auto bias = m.insert_instruction(
                add, make_op("multibroadcast", {{"out_lens", bias_lens}}), bcast->inputs().front());
            if(not std::is_sorted(perm.begin(), perm.end()))
                bias =
                    m.insert_instruction(add, make_op("transpose", {{"permutation", perm}}), bias);
            bias = m.insert_instruction(
                add, make_op("reshape", {{"dims", {result->get_shape().lens()[1]}}}), bias);
            bias = m.insert_instruction(
                add,
                make_op("broadcast", {{"axis", 1}, {"out_lens", result->get_shape().lens()}}),
                bias);
            result = m.insert_instruction(add, make_op("add"), result, bias);
            last   = add;
        }
        result = m.insert_instruction(last, make_op("reshape", {{"dims", split_lens}}), result);
        if(not std::is_sorted(perm.begin(), perm.end()))
            result = m.insert_instruction(
                last, make_op("transpose", {{"permutation", invert_permutation(perm)}}), result);
        m.replace_instruction(last, result);
    }
};

} // namespace

void rewrite_nd_convolution::apply(module& m) const
{
    match::find_matches(m, find_nd_convolution{max_spatial_dims});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
