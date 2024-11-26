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
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

struct find_dot
{
    auto matcher() const { return match::name("dot"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins     = r.result;
        auto a_mat   = ins->inputs().front();
        auto b_mat   = ins->inputs().back();
        auto a_shape = a_mat->get_shape();
        auto b_shape = b_mat->get_shape();
        auto ndim    = a_shape.ndim();
        auto batch   = std::accumulate(
            a_shape.lens().begin(), a_shape.lens().end() - 2, 1, std::multiplies<>{});
        if(batch != 1)
            return;
        auto rows = a_shape.lens().at(ndim - 2);
        if(rows > 2)
            return;

        // If the b matrix is const foldable then make sure its a transposed layout unless its
        // broadcasting
        if(b_mat->can_eval() and not b_shape.transposed())
        {
            std::vector<int64_t> permutation(ndim);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::swap(permutation.back(), permutation.at(ndim - 2));
            b_mat =
                m.insert_instruction(ins, make_op("layout", {{"permutation", permutation}}), b_mat);
        }

        auto a_unsqueeze =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {ndim}}}), a_mat);
        auto b_unsqueeze =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {ndim - 2}}}), b_mat);
        auto mul    = insert_common_op(m, ins, make_op("mul"), {a_unsqueeze, b_unsqueeze});
        auto reduce = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", {ndim - 1}}}), mul);
        m.replace_instruction(ins, make_op("squeeze", {{"axes", {ndim - 1}}}), reduce);
    }
};

struct find_softmax
{
    auto matcher() const { return match::name("softmax"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto op   = ins->get_operator().to_value();
        auto axis = op["axis"].to<std::int64_t>();

        auto input = ins->inputs().front();
        auto max   = m.insert_instruction(ins, make_op("reduce_max", {{"axes", {axis}}}), input);
        auto maxb  = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", input->get_shape().lens()}}), max);
        auto sub  = m.insert_instruction(ins, make_op("sub"), input, maxb);
        auto exp  = m.insert_instruction(ins, make_op("exp"), sub);
        auto sum  = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", {axis}}}), exp);
        auto sumb = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", input->get_shape().lens()}}), sum);
        m.replace_instruction(ins, make_op("div"), exp, sumb);
    }
};

struct find_reduce_mean_variance
{
    auto matcher() const
    {
        auto reduce_mean = match::name("reduce_mean");
        auto skip_broadcasts_mean = match::skip_broadcasts(reduce_mean.bind("mean"));
        auto x_minus_mean         = match::name("sub")(match::arg(0)(match::any().bind("x")),
                                               match::arg(1)(skip_broadcasts_mean));
        auto pow_x_minus_mean =
            match::name("pow")(match::arg(0)(x_minus_mean), match::arg(1)(match::has_value(2.0f)));
        auto mul_x_minus_mean =
            match::name("mul")(match::arg(0)(x_minus_mean), match::arg(1)(x_minus_mean));
        auto sqdiff = match::name("sqdiff")(
            match::either_arg(0, 1)(match::any().bind("x"), skip_broadcasts_mean));
        return reduce_mean(
            match::arg(0)(match::any_of(pow_x_minus_mean, mul_x_minus_mean, sqdiff)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto mean  = r.instructions["mean"];

        if(ins->get_operator() != mean->get_operator())
            return;

        if(mean->inputs().front() != x_ins)
            return;

        auto x2       = m.insert_instruction(ins, make_op("mul"), x_ins, x_ins);
        auto mean_x2  = m.insert_instruction(ins, mean->get_operator(), x2);
        auto mean_x_2 = m.insert_instruction(ins, make_op("mul"), mean, mean);
        m.replace_instruction(ins, make_op("sub"), mean_x2, mean_x_2);
    }
};

struct find_reduce_mean
{
    auto matcher() const { return match::name("reduce_mean"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto op    = ins->get_operator().to_value();
        auto axes  = op["axes"].to_vector<std::int64_t>();
        auto input = ins->inputs().front();

        bool is_integral = false;
        double max_n     = 0;
        std::size_t size = 0;
        input->get_shape().visit_type([&](auto t) {
            is_integral = t.is_integral();
            max_n       = t.max();
            size        = t.size();
        });

        auto n = input->get_shape().elements() / ins->get_shape().elements();

        // Convert accumulator to float if <= 8bit type or if < 3 bytes and n >= max_n /4
        if(size == 1 or (n >= max_n / 4 and size < 3))
        {
            shape::type_t t = is_integral ? shape::int32_type : shape::float_type;
            input = m.insert_instruction(ins, make_op("convert", {{"target_type", t}}), input);
        }

        auto n_literal = m.add_literal(literal{{input->get_shape().type(), {1}}, {n}});
        if(is_integral)
        {
            auto reduce_sum =
                m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), input);
            auto div = insert_common_op(m, ins, make_op("div"), {reduce_sum, n_literal});
            m.replace_instruction(
                ins, make_op("convert", {{"target_type", ins->get_shape().type()}}), div);
        }
        else
        {
            auto new_input = insert_common_op(m, ins, make_op("div"), {input, n_literal});
            auto reduce_sum =
                m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), new_input);
            m.replace_instruction(
                ins, make_op("convert", {{"target_type", ins->get_shape().type()}}), reduce_sum);
        }
    }
};

} // namespace

void rewrite_reduce::apply(module& m) const
{
    match::find_matches(m, find_dot{});
    match::find_matches(m, find_softmax{}, find_reduce_mean_variance{});
    match::find_matches(m, find_reduce_mean{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
