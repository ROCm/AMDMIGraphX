/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/rewrite_dot.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>

#include <migraphx/check_shapes.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_REWRITE_DOT);

struct window
{
    std::vector<int64_t> axes    = {};
    std::vector<int64_t> stride  = {};
    std::vector<int64_t> lengths = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.stride, "stride"), f(self.lengths, "lengths"));
    }

    std::string name() const { return "window"; }

    shape compute_shape(const std::vector<shape>& inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        const auto& input = inputs[0];
        auto lens         = input.lens();
        auto strides      = input.strides();
        for(auto i : range(axes.size()))
        {
            auto axis       = axes[i];
            auto win_stride = stride[i];
            auto win_len    = lengths[i];
            auto& dim       = lens[axis];
            auto& s         = strides[axis];
            dim             = ((dim - win_len) / win_stride) + 1;
            s *= win_stride;
        }
        std::copy(lengths.begin(), lengths.end(), std::back_inserter(lens));
        std::transform(axes.begin(), axes.end(), std::back_inserter(strides), [&](auto axis) {
            return input.strides()[axis];
        });
        shape result{input.type(), lens, strides};
        if(result.element_space() > input.element_space())
            MIGRAPHX_THROW("Out of bounds window access");
        return result;
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        return args[0].reshape(output_shape);
    }
};
MIGRAPHX_REGISTER_OP(window);

namespace {

MIGRAPHX_PRED_MATCHER(conv_1x1, instruction_ref ins)
{
    if(ins->name() != "convolution")
        return false;
    auto v = ins->get_operator().to_value();
    if(not all_of(v.at("stride"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    if(not all_of(v.at("padding"), [](const value& x) { return x.to<std::size_t>() == 0; }))
        return false;
    if(not all_of(v.at("dilation"), [](const value& x) { return x.to<std::size_t>() == 1; }))
        return false;
    auto w = ins->inputs().at(1)->get_shape();
    return std::all_of(w.lens().begin() + 2, w.lens().end(), [](std::size_t i) { return i == 1; });
}

std::vector<int64_t> nhwc_perm(std::size_t ndim)
{
    std::vector<int64_t> result(ndim);
    std::iota(result.begin(), result.end(), 0);
    std::rotate(result.begin() + 1, result.begin() + 2, result.end());
    return result;
}

struct find_1x1_convolution
{
    auto matcher() const { return conv_1x1(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();
        auto m_dim   = std::accumulate(input->get_shape().lens().begin() + 2,
                                     input->get_shape().lens().end(),
                                     input->get_shape().lens().front(),
                                     std::multiplies<>{});
        auto n_dim   = weights->get_shape().lens()[0];
        auto k_dim   = weights->get_shape().lens()[1];

        std::vector<int64_t> aperm = nhwc_perm(ins->get_shape().ndim());
        auto transpose =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", aperm}}), input);
        auto a_mat =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {m_dim, k_dim}}}), transpose);

        auto reshape =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {n_dim, k_dim}}}), weights);
        auto b_mat =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {1, 0}}}), reshape);

        auto dot        = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
        auto out_dims   = transpose->get_shape().lens();
        out_dims.back() = n_dim;
        auto creshape   = m.insert_instruction(ins, make_op("reshape", {{"dims", out_dims}}), dot);
        m.replace_instruction(
            ins, make_op("transpose", {{"permutation", invert_permutation(aperm)}}), creshape);
    }
};

struct find_all_convolution
{
    auto matcher() const { return match::name("convolution"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto kdims = ins->get_shape().ndim() - 2;
        auto input   = ins->inputs().front();
        auto weights = ins->inputs().back();

        auto v = ins->get_operator().to_value();
        auto vpadding = v.at("padding").to_vector<std::size_t>();
        auto vstride = v.at("stride").to_vector<std::size_t>();

        std::vector<std::size_t> padding(2);
        padding.insert(padding.end(), vpadding.begin(), vpadding.begin() + kdims);
        padding.push_back(0);
        padding.push_back(0);
        if(vpadding.size() > kdims)
            padding.insert(padding.end(), vpadding.begin() + kdims, vpadding.end());
        else
            padding.insert(padding.end(), vpadding.begin(), vpadding.begin() + kdims);

        auto pad = m.insert_instruction(ins, make_op("pad", {{"pads", padding}}), input);

        window w;
        w.axes.resize(kdims);
        std::iota(w.axes.begin(), w.axes.end(), 2);
        w.lengths.insert(w.lengths.end(), weights->get_shape().lens().begin() + 2, weights->get_shape().lens().end());
        w.stride.insert(w.stride.end(), vstride.begin(), vstride.end());

        auto iwindow = m.insert_instruction(ins, w, pad);

        auto out_channels   = weights->get_shape().lens()[0];
        auto in_channels   = weights->get_shape().lens()[1];

        auto m_dim   = std::accumulate(iwindow->get_shape().lens().begin() + 2,
                                     iwindow->get_shape().lens().begin() + 2 + kdims,
                                     iwindow->get_shape().lens().front(),
                                     std::multiplies<>{});
        auto k_dim   = std::accumulate(weights->get_shape().lens().begin() + 2, weights->get_shape().lens().end(), in_channels);
        auto n_dim   = out_channels;

        std::vector<int64_t> aperm(iwindow->get_shape().ndim());
        std::iota(aperm.begin(), aperm.end(), 0);
        std::rotate(aperm.begin() + 1, aperm.begin() + 2, aperm.begin() + 2 + kdims);
        auto transpose =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", aperm}}), iwindow);
        auto a_mat =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {m_dim, k_dim}}}), transpose);

        auto reshape =
            m.insert_instruction(ins, make_op("reshape", {{"dims", {n_dim, k_dim}}}), weights);
        auto b_mat =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {1, 0}}}), reshape);

        auto dot        = m.insert_instruction(ins, make_op("dot"), a_mat, b_mat);
        auto bperm = nhwc_perm(ins->get_shape().ndim());
        auto out_dims   = reorder_dims(ins->get_shape().lens(), bperm);

        auto creshape   = m.insert_instruction(ins, make_op("reshape", {{"dims", out_dims}}), dot);
        m.replace_instruction(
            ins, make_op("transpose", {{"permutation", invert_permutation(bperm)}}), creshape);
    }
};

} // namespace

void rewrite_dot::apply(module& m) const 
{
    if(not enabled(MIGRAPHX_ENABLE_REWRITE_DOT{}))
        return;
    match::find_matches(m, find_1x1_convolution{});
    match::find_matches(m, find_all_convolution{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
