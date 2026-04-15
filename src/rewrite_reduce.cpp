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
#include <migraphx/rewrite_reduce.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/match/softmax.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/common.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <migraphx/unfold.hpp>
#include <migraphx/dead_code_elimination.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DISABLE_FP32_SOFTMAX);

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

        std::vector<int64_t> permutation(ndim);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::swap(permutation.back(), permutation.at(ndim - 2));

        // If the b matrix is const foldable then make sure its a transposed layout unless its
        // broadcasting
        if(b_mat->can_eval() and not b_shape.transposed())
        {
            b_mat =
                m.insert_instruction(ins, make_op("layout", {{"permutation", permutation}}), b_mat);
        }

        auto a_unsqueeze =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {ndim - 1}}}), a_mat);
        auto b_transpose =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", permutation}}), b_mat);
        auto b_unsqueeze =
            m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {ndim - 2}}}), b_transpose);
        auto mul    = insert_common_op(m, ins, make_op("mul"), {a_unsqueeze, b_unsqueeze});
        auto reduce = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", {ndim}}}), mul);
        m.replace_instruction(ins, make_op("squeeze", {{"axes", {ndim}}}), reduce);
    }
};

struct find_logsoftmax
{
    auto matcher() const { return match::name("logsoftmax"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto op   = ins->get_operator().to_value();
        auto axis = op["axis"].to<std::int64_t>();

        auto input   = ins->inputs().front();
        auto softmax = m.insert_instruction(ins, make_op("softmax", {{"axis", axis}}), input);
        m.replace_instruction(ins, make_op("log"), softmax);
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

// Extend the FP32 upcast range from the dot output through mul/where to
// softmax. Prevents FP16 overflow in Q*K attention dot products for models
// with large k_proj.bias values (e.g. Qwen, DeepSeek).
//
// The dot stays as dot(f16,f16)->f16. A convert(f16->f32) is inserted on
// its output, and the intermediate ops (mul, where) are upcasted to f32.
// MFMA/WMMA accumulates in f32 internally; when fused into an attention
// kernel, rocMLIR's RemoveRedundantCasts pass preserves the f32 accumulator.
//
// Runs before find_softmax_base_ops so that the softmax internals
// (reduce_max through div) are still in f16 when find_softmax_base_ops
// processes them.
struct find_dot_softmax_fp32
{
    auto matcher() const { return match::softmax(); }

    // Walk backwards from the softmax input through the attention chain
    // to find an upstream dot. At each step, follows the non-constant,
    // non-bool input (the attention data path), skipping constants (scale,
    // -inf literals) and bool inputs (where conditions/masks).
    static std::optional<instruction_ref> find_upstream_dot(instruction_ref inp)
    {
        auto step = [](instruction_ref current) -> std::optional<instruction_ref> {
            if(current->name() == "dot")
                return std::nullopt;
            if(current->inputs().size() == 1)
                return current->inputs().front();
            auto it = std::find_if(
                current->inputs().begin(), current->inputs().end(), [](instruction_ref input) {
                    return not input->can_eval() and input->get_shape().type() != shape::bool_type;
                });
            if(it == current->inputs().end())
                return std::nullopt;
            return *it;
        };
        auto chain = unfold(inp, step);
        auto it    = std::find_if(
            chain.begin(), chain.end(), [](instruction_ref ins) { return ins->name() == "dot"; });
        if(it != chain.end())
            return *it;
        return std::nullopt;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto inp      = r.instructions["x"];
        auto inp_type = inp->get_shape().type();

        if(contains({shape::float_type, shape::double_type}, inp_type))
            return;

        auto dot_opt = find_upstream_dot(inp);
        if(not dot_opt.has_value())
            return;

        // Upcast ops between dot (exclusive) and inp (inclusive)
        auto dot_ins  = *dot_opt;
        auto pre_inss = find_instructions_between(dot_ins, inp, &m);

        for(const auto& ins : pre_inss)
        {
            if(ins == dot_ins)
                continue;

            std::vector<instruction_ref> ins_inputs_up;
            std::transform(
                ins->inputs().begin(),
                ins->inputs().end(),
                std::back_inserter(ins_inputs_up),
                [&](auto i) {
                    if(i->get_shape().type() == shape::bool_type or
                       i->get_shape().type() == shape::float_type)
                        return i;
                    return m.insert_instruction(
                        ins, make_op("convert", {{"target_type", shape::float_type}}), i);
                });

            auto ins_up = m.insert_instruction(ins, ins->get_operator(), ins_inputs_up);
            m.replace_instruction(
                ins, make_op("convert", {{"target_type", ins->get_shape().type()}}), ins_up);
        }
    }
};

struct find_softmax_base_ops
{
    bool full_precision;

    auto matcher() const { return match::softmax(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto div             = r.result;
        auto inp             = r.instructions["x"];
        auto inp_type        = inp->get_shape().type();
        auto requires_upcast = not contains({shape::float_type, shape::double_type}, inp_type);

        if(not requires_upcast)
            return;

        auto softmax_inss = find_instructions_between(inp, div, &m);

        for(const auto& ins : softmax_inss)
        {
            if(ins == inp)
                continue;

            // Upcast inputs
            std::vector<instruction_ref> ins_inputs_up;
            std::transform(
                ins->inputs().begin(),
                ins->inputs().end(),
                std::back_inserter(ins_inputs_up),
                [&](auto i) {
                    return m.insert_instruction(
                        ins, make_op("convert", {{"target_type", shape::float_type}}), i);
                });

            // Duplicate instruction to perform op in higher precision
            auto ins_up = m.insert_instruction(ins, ins->get_operator(), ins_inputs_up);

            // replace original ins with downcast to preserve graph validity
            m.replace_instruction(
                ins, make_op("convert", {{"target_type", ins->get_shape().type()}}), ins_up);
        }
    }
};

struct find_reduce_mean_variance
{
    auto matcher() const
    {
        auto reduce_mean          = match::name("reduce_mean");
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
    match::find_matches(m, find_logsoftmax{});
    match::find_matches(m, find_softmax{}, find_reduce_mean_variance{});

    if(not enabled(MIGRAPHX_DISABLE_FP32_SOFTMAX{}))
    {
        match::find_matches(m, find_dot_softmax_fp32{});
        match::find_matches(m, find_softmax_base_ops{});
        migraphx::run_passes(m,
                             {migraphx::eliminate_convert{},
                              migraphx::dead_code_elimination{},
                              migraphx::eliminate_common_subexpression{}});
    }

    match::find_matches(m, find_reduce_mean{});
    migraphx::run_passes(m, {simplify_reshapes{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
