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
#include <migraphx/gpu/prepare_mlir.hpp>
#include <migraphx/common.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape_for_each.hpp>
#include <algorithm>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

struct find_reduce
{
    auto matcher() const { return match::name_contains("reduce"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins            = r.result;
        auto reduce_op      = ins->get_operator().to_value();
        auto reduce_op_name = ins->get_operator().name();
        auto reduce_axes    = reduce_op["axes"].to_vector<size_t>();
        auto reduce_lens    = ins->get_shape().lens();
        auto in_shape       = ins->inputs().front()->get_shape();
        const auto& in_lens = in_shape.lens();
        assert(in_shape.standard());
        assert(reduce_lens.size() == in_lens.size());
        assert(std::adjacent_find(
                   reduce_axes.begin(), reduce_axes.end(), [](auto axis_1, auto axis_2) {
                       return axis_2 - axis_1 > 1;
                   }) == reduce_axes.end());

        std::vector<int64_t> new_rsp_dims;
        std::vector<int64_t> new_reduce_axes;
        for(const auto axis : range(in_shape.ndim()))
        {
            if(reduce_lens[axis] == in_lens[axis])
            {
                new_rsp_dims.push_back(in_lens[axis]);
            }
            else if(new_reduce_axes.empty())
            {
                assert(reduce_lens[axis] == 1);
                new_rsp_dims.push_back(-1);
                new_reduce_axes.push_back(axis);
            }
        }
        auto rsp_ins = m.insert_instruction(
            ins, migraphx::make_op("reshape", {{"dims", new_rsp_dims}}), ins->inputs().front());
        auto collapsed_reduce = m.insert_instruction(
            ins, migraphx::make_op(reduce_op_name, {{"axes", new_reduce_axes}}), rsp_ins);
        auto rsp_back = m.insert_instruction(
            ins, migraphx::make_op("reshape", {{"dims", reduce_lens}}), collapsed_reduce);
        m.replace_instruction(ins, rsp_back);
    }
};

struct find_leaky_relu
{
    auto matcher() const { return match::name("leaky_relu"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto x_ins = ins->inputs().front();

        float alpha_f = ins->get_operator().to_value()["alpha"].to<float>();
        auto alpha     = m.add_literal(literal{{x_ins->get_shape().type(), {1}}, {alpha_f}});
        auto mul_alpha = insert_common_op(m, ins, make_op("mul"), {x_ins, alpha});
        if(alpha_f >= 0.0f and alpha_f <= 1.0f)
        {
            auto max_ins = insert_common_op(m, ins, make_op("max"), {x_ins, mul_alpha});
            m.replace_instruction(ins, max_ins);
        }
        else
        {
            auto zero    = m.add_literal(literal{{x_ins->get_shape().type(), {1}}, {0.0}});
            auto greater = insert_common_op(m, ins, make_op("greater"), {x_ins, zero});
            m.replace_instruction(ins, make_op("where"), {greater, x_ins, mul_alpha});
        }
    }
};

// rocMLIR resolves the kv-cache sequence length by unwrapping a single
// broadcast from the mask compare, and requires one entry per gemm batch once
// the attention heads are folded into it. The sequence length input holds one
// value per batch (concat_past_present indexes it by batch), so the trace ends
// at a {batch}-element tensor, which fails to verify. Broadcast the sequence
// length over the heads dimension in a separate step so rocMLIR binds a
// {batch, heads} tensor it can collapse to match the attention batch.
struct find_kv_cache_mask_seq_len
{
    static const std::unordered_set<std::string>& view_ops()
    {
        static const std::unordered_set<std::string> names = {
            "multibroadcast", "broadcast", "reshape", "unsqueeze", "squeeze"};
        return names;
    }

    auto matcher() const
    {
        auto seq_len = match::skip(match::name(view_ops()))(match::name("@param").bind("seq_len"));
        auto compare =
            match::name("greater")(match::used_once(), match::arg(1)(seq_len)).bind("greater");
        auto cond = match::skip(match::name(
            "convert", "multibroadcast", "broadcast", "reshape", "unsqueeze", "squeeze"))(compare);
        return match::name("where")(match::arg(0)(cond));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto where_ins = r.result;
        auto greater   = r.instructions["greater"];
        auto seq_len   = r.instructions["seq_len"];

        if(seq_len->get_shape().type() != shape::int32_type)
            return;
        const auto& mask_lens = where_ins->get_shape().lens();
        if(mask_lens.size() != 4)
            return;
        std::vector<std::size_t> lead_lens(mask_lens.begin(), mask_lens.end() - 2);
        // One sequence length per batch, with heads to broadcast it over
        if(seq_len->get_shape().elements() != lead_lens[0])
            return;
        if(lead_lens[1] == 1)
            return;
        // Already rewritten if the sequence length is broadcast over the
        // leading dimensions in a separate step
        for(auto ins = greater->inputs().back(); ins != seq_len; ins = ins->inputs().front())
        {
            if(ins->get_shape().lens() == lead_lens)
                return;
        }
        // Only handle the column indices as a single broadcast since rocMLIR
        // only unwraps one broadcast to find the constant range
        auto col_bcast = greater->inputs().front();
        if(col_bcast->name() != "multibroadcast")
            return;
        auto col_ins         = col_bcast->inputs().front();
        const auto& col_lens = col_ins->get_shape().lens();
        if(col_lens.size() > mask_lens.size())
            return;
        if(not std::equal(
               col_lens.rbegin(), col_lens.rend(), mask_lens.rbegin(), [](auto len, auto mask_len) {
                   return len == mask_len or len == 1;
               }))
            return;

        auto new_col = m.insert_instruction(
            greater, make_op("multibroadcast", {{"out_lens", mask_lens}}), col_ins);
        std::vector<std::size_t> batch_lens{lead_lens[0], 1};
        auto flat =
            m.insert_instruction(greater, make_op("reshape", {{"dims", batch_lens}}), seq_len);
        auto lead = m.insert_instruction(
            greater, make_op("multibroadcast", {{"out_lens", lead_lens}}), flat);
        auto unsq = m.insert_instruction(greater, make_op("unsqueeze", {{"axes", {2, 3}}}), lead);
        auto new_seq_len = m.insert_instruction(
            greater, make_op("multibroadcast", {{"out_lens", mask_lens}}), unsq);
        auto new_greater =
            m.insert_instruction(greater, greater->get_operator(), {new_col, new_seq_len});
        m.replace_instruction(greater, new_greater);
    }
};

// rocmlirTriton rebuilds the per-group kv-cache sequence length from the
// collapse_shape that feeds Q into the first gemm. When Q is a plain parameter
// that collapse folds into the parameter's own reshape, the sequence length is
// passed through unbroadcast and the kernel reads it out of bounds for every
// group after the first. Give a decode-shaped Q a transpose producer that only
// moves its unit sequence dimension: it costs nothing and keeps the collapse.
struct find_kv_cache_plain_q
{
    auto matcher() const
    {
        auto seq_len =
            match::skip(match::name(find_kv_cache_mask_seq_len::view_ops()))(match::name("@param"));
        auto compare = match::name("greater")(match::arg(1)(seq_len));
        auto cond    = match::skip(match::name(
            "convert", "multibroadcast", "broadcast", "reshape", "unsqueeze", "squeeze"))(compare);
        return match::name("where")(match::arg(0)(cond));
    }

    // The masked scores are the first gemm, possibly scaled, converted, or
    // wrapped by an earlier mask
    static std::optional<instruction_ref> find_scores_gemm(instruction_ref ins)
    {
        while(ins->name() != "dot")
        {
            const auto& inputs = ins->inputs();
            if(ins->name() == "where")
            {
                ins = inputs.at(2);
                continue;
            }
            if(not ins->get_operator().attributes().get("pointwise", false))
                return std::nullopt;
            auto data = std::find_if(inputs.begin(), inputs.end(), [](instruction_ref input) {
                return not input->get_shape().broadcasted() and not input->can_eval();
            });
            if(data == inputs.end())
                return std::nullopt;
            ins = *data;
        }
        return ins;
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto gemm = find_scores_gemm(r.result);
        if(not gemm.has_value())
            return;
        auto q = (*gemm)->inputs().front();
        if(q->name() != "@param")
            return;
        // Only {batch, heads, 1, dim} needs the per-head broadcast and can swap
        // the unit dimension for free
        const auto& lens = q->get_shape().lens();
        if(lens.size() != 4 or lens[1] == 1 or lens[2] != 1)
            return;
        std::vector<std::size_t> swapped{lens[0], lens[2], lens[1], lens[3]};
        auto rsp = m.insert_instruction(*gemm, make_op("reshape", {{"dims", swapped}}), q);
        auto tsp =
            m.insert_instruction(*gemm, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), rsp);
        m.replace_instruction(*gemm, (*gemm)->get_operator(), {tsp, (*gemm)->inputs().back()});
    }
};

// True when the literal is a square boolean mask with ones strictly above
// the diagonal (the causal mask emitted by e.g. the linear attention parser)
bool is_upper_triangle_mask(const literal& lit)
{
    const auto& s = lit.get_shape();
    if(s.type() != shape::bool_type or s.ndim() != 2 or s.lens().front() != s.lens().back())
        return false;
    bool result = true;
    lit.visit([&](auto v) {
        shape_for_each(s, [&](const auto& idx, std::size_t i) {
            if(float_equal(v[i], 0) == (idx[1] > idx[0]))
                result = false;
        });
    });
    return result;
}

// True when the literal is a square numeric 0/1 mask with ones on and below
// the diagonal: multiplying scores by it zeroes the strictly-upper triangle
bool is_lower_triangle_mask(const literal& lit)
{
    const auto& s = lit.get_shape();
    if(s.type() == shape::bool_type or s.ndim() != 2 or s.lens().front() != s.lens().back())
        return false;
    bool result = true;
    lit.visit([&](auto v) {
        shape_for_each(s, [&](const auto& idx, std::size_t i) {
            if(not float_equal(v[i], idx[1] > idx[0] ? 0 : 1))
                result = false;
        });
    });
    return result;
}

// Insert a (t, t) boolean predicate that is true strictly above the diagonal,
// as a compare of index iotas that the broadcast sink turns into the form
// rocMLIR recognizes as the kernel's causal mode
instruction_ref insert_causal_mask_pred(module& m, instruction_ref pos, std::size_t t)
{
    std::vector<std::size_t> mask_lens{t, t};
    std::vector<std::int32_t> iota_vals(t);
    std::iota(iota_vals.begin(), iota_vals.end(), 0);
    auto iota_lit = m.add_literal(literal{shape{shape::int32_type, {t}}, iota_vals});
    auto rows     = m.insert_instruction(pos, make_op("unsqueeze", {{"axes", {1}}}), iota_lit);
    auto rows_b =
        m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", mask_lens}}), rows);
    auto cols_b =
        m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", mask_lens}}), iota_lit);
    auto gt = m.insert_instruction(pos, make_op("greater"), cols_b, rows_b);
    return m.insert_instruction(pos, make_op("convert", {{"target_type", shape::bool_type}}), gt);
}

// A causal mask literal (inlined by fuse_mlir) is opaque data once it reaches
// rocMLIR: replace it with the element-wise equal compare of index iotas — a
// bool zero-mask becomes the predicate itself, a multiplicative 0/1 mask
// becomes where(predicate, 0, 1) for the mul fold below to consume.
struct find_causal_mask_literal
{
    auto matcher() const { return match::name("@literal"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins        = r.result;
        const auto& lit = ins->get_literal();
        bool upper      = is_upper_triangle_mask(lit);
        if(not upper and not is_lower_triangle_mask(lit))
            return;
        auto t    = ins->get_shape().lens().front();
        auto pos  = std::next(ins);
        auto pred = insert_causal_mask_pred(m, pos, t);
        if(upper)
        {
            m.replace_instruction(ins, pred);
            return;
        }
        std::vector<std::size_t> mask_lens{t, t};
        auto type = ins->get_shape().type();
        auto zero = m.add_literal(literal{shape{type, {1}}, {0.0}});
        auto one  = m.add_literal(literal{shape{type, {1}}, {1.0}});
        auto zero_b =
            m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", mask_lens}}), zero);
        auto one_b =
            m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", mask_lens}}), one);
        auto mask = m.insert_instruction(pos, make_op("where"), {pred, zero_b, one_b});
        m.replace_instruction(ins, mask);
    }
};

// Multiplying by a 0/1 mask of the form where(pred, 0, 1) keeps the values
// the predicate rejects: fold the mul into the select, mul(x, where(p, 0, 1))
// == where(p, 0, x), so a causal mask compare feeds the select directly and
// rocMLIR can lower it as the kernel's causal mode.
struct find_masked_mul
{
    auto matcher() const
    {
        auto mask = match::name("where")(match::arg(1)(match::has_value(0.0f)),
                                         match::arg(2)(match::has_value(1.0f)));
        return match::name("mul")(match::either_arg(0, 1)(match::skip_broadcasts(mask.bind("mask")),
                                                          match::any().bind("x")));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto mask        = r.instructions["mask"];
        auto x           = r.instructions["x"];
        const auto& lens = ins->get_shape().lens();
        auto pred        = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", lens}}), mask->inputs().front());
        auto zero = m.add_literal(literal{{ins->get_shape().type(), {1}}, {0.0}});
        auto zero_b =
            m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", lens}}), zero);
        m.replace_instruction(ins, make_op("where"), {pred, zero_b, x});
    }
};

// rocMLIR recognizes a causal mask structurally as a select whose predicate
// is a direct compare of constant index ranges, but a mask computed at its
// own shape reaches the select through a broadcast that hides the compare.
// Sink the broadcast below the compare (and its convert) by re-broadcasting
// the compare's operands to the outer shape.
struct find_broadcast_of_compare
{
    auto matcher() const
    {
        auto compare   = match::name("greater", "less")(
                             match::all_of[match::inputs()](match::name("multibroadcast")))
                             .bind("compare");
        auto converted = match::name("convert")(match::arg(0)(compare));
        return match::name("multibroadcast")(match::arg(0)(match::any_of(compare, converted)));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins         = r.result;
        auto input       = ins->inputs().front();
        auto compare     = r.instructions["compare"];
        const auto& lens = ins->get_shape().lens();
        std::vector<instruction_ref> new_args;
        std::transform(
            compare->inputs().begin(),
            compare->inputs().end(),
            std::back_inserter(new_args),
            [&](instruction_ref b) {
                return m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", lens}}), b->inputs().front());
            });
        auto sunk = m.insert_instruction(ins, compare->get_operator(), new_args);
        if(input != compare)
            sunk = m.insert_instruction(ins, input->get_operator(), {sunk});
        m.replace_instruction(ins, sunk);
    }
};

// mlir has issues sometime when the condition to `where` is not a bool. So this will convert the
// condition to a bool.
struct find_where
{
    auto matcher() const { return match::name("where"); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins      = r.result;
        auto cond_ins = ins->inputs().front();

        if(cond_ins->get_shape().type() == shape::bool_type)
            return;

        auto bool_cond_ins = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), cond_ins);

        m.replace_instruction(
            ins, make_op("where"), {bool_cond_ins, ins->inputs()[1], ins->inputs()[2]});
    }
};

// mlir requires literals to be in a standard shape.
struct find_nonstandard_literal
{
    auto matcher() const
    {
        return match::name("@literal")(match::not_standard_shape(),
                                       match::none_of(match::broadcast_shape()));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto arg = ins->get_literal().get_argument();
        shape s{arg.get_shape().type(), arg.get_shape().lens()};
        literal result;
        visit_all(arg)([&](auto x) { result = literal{s, x.to_vector()}; });
        m.replace_instruction(ins, m.add_literal(result));
    }
};

} // namespace

bool is_causal_mask_literal(const literal& lit)
{
    return is_upper_triangle_mask(lit) or is_lower_triangle_mask(lit);
}

void prepare_mlir::apply(module& m) const
{
    match::find_matches(m, find_reduce{}, find_leaky_relu{}, find_kv_cache_mask_seq_len{});
    // Matches the same where as find_kv_cache_mask_seq_len, so it needs its own pass
    match::find_matches(m, find_kv_cache_plain_q{});
    // Literals precede their consumers, so the mask rewrite is visible to the
    // mul fold in the same pass; both run before the broadcast sink so the
    // compares they create are canonicalized by the next pass
    match::find_matches(m, find_causal_mask_literal{}, find_masked_mul{});
    match::find_matches(m, find_broadcast_of_compare{}, find_where{}, find_nonstandard_literal{});
    run_passes(m, {dead_code_elimination{}});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
