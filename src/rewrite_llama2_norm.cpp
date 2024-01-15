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
 */

#include <migraphx/rewrite_llama2_norm.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_llama2_norm
{
    auto pow() const
    {
        return match::name("pow")(match::arg(0)(match::any().bind("y"))).bind("pow");
    }

    auto last_axis() const
    {
        return match::make_basic_pred_matcher([](instruction_ref ins) {
            auto v = ins->get_operator().to_value();
            if(not v.contains("axes"))
                return false;
            auto axes = v["axes"].to_vector<std::size_t>();
            if(axes.size() != 1)
                return false;
            return axes.front() == ins->inputs().front()->get_shape().lens().size() - 1;
        });
    }

    auto reduce_mean() const
    {
        return match::name("reduce_mean")(last_axis()(match::arg(0)(pow()))).bind("reduce_mean");
    }

    auto add() const
    {
        return match::name("add")(match::either_arg(0, 1)(match::is_constant(), reduce_mean()))
            .bind("add");
    }

    auto sqrt() const { return match::name("sqrt")(match::arg(0)(add())).bind("sqrt"); }

    auto div() const
    {
        return match::name("div")(match::arg(0)(match::has_value(1.0f)), match::arg(1)(sqrt()))
            .bind("div");
    }

    auto mbcast() const
    {
        return match::name("multibroadcast")(match::arg(0)(div()).bind("mbcast"));
    }

    auto contig() const
    {
        return match::name("contiguous")(match::arg(0)(mbcast()).bind("contig"));
    }

    auto mul() const
    {
        return match::name("mul")(match::either_arg(0, 1)(match::any().bind("x"),
                                                          match::any_of(contig(), mbcast())))
            .bind("mul");
    }

    auto matcher() const { return mul(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        // We only need to run when fp16 is used
        auto x = r.instructions["x"];
        if(x->get_shape().type() != migraphx::shape::half_type)
            return;

        auto pow = r.instructions["pow"];
        std::vector<instruction_ref> pow_inputs;
        pow_inputs.emplace_back(m.insert_instruction(
            pow, make_op("convert", {{"target_type", shape::float_type}}), pow->inputs().at(0)));
        pow_inputs.emplace_back(m.insert_instruction(
            pow, make_op("convert", {{"target_type", shape::float_type}}), pow->inputs().at(1)));
        auto converted_pow = m.insert_instruction(pow, pow->get_operator(), pow_inputs);

        auto reduce_mean = r.instructions["reduce_mean"];
        auto converted_reduce_mean =
            m.insert_instruction(reduce_mean, reduce_mean->get_operator(), converted_pow);

        auto add = r.instructions["add"];
        std::vector<instruction_ref> add_inputs;
        add_inputs.emplace_back(converted_reduce_mean);
        add_inputs.emplace_back(m.insert_instruction(
            add, make_op("convert", {{"target_type", shape::float_type}}), add->inputs().at(1)));
        auto converted_add = m.insert_instruction(add, add->get_operator(), add_inputs);

        auto sqrt           = r.instructions["sqrt"];
        auto converted_sqrt = m.insert_instruction(sqrt, sqrt->get_operator(), converted_add);

        auto div = r.instructions["div"];
        std::vector<instruction_ref> div_inputs;
        div_inputs.emplace_back(m.insert_instruction(
            div, make_op("convert", {{"target_type", shape::float_type}}), div->inputs().at(0)));
        div_inputs.emplace_back(converted_sqrt);
        auto converted_div = m.insert_instruction(div, div->get_operator(), div_inputs);

        auto mbcast = r.instructions["mbcast"];
        std::vector<instruction_ref> mbcast_inputs;
        mbcast_inputs.emplace_back(m.insert_instruction(
            mbcast, make_op("convert", {{"target_type", shape::half_type}}), converted_div));
        auto converted_mbcast = m.insert_instruction(mbcast, mbcast->get_operator(), mbcast_inputs);

        m.replace_instruction(mbcast, converted_mbcast);
    }
};

void rewrite_llama2_norm::apply(module& m) const { match::find_matches(m, find_llama2_norm{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
