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
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <numeric>
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
// the attention heads are folded into it. When the sequence length is a
// scalar input the trace ends at a one-element tensor, which fails to
// verify. Broadcast the sequence length over the leading batch and heads
// dimensions in a separate step so rocMLIR binds a {batch, heads} tensor it
// can collapse to match the attention batch.
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
        if(seq_len->get_shape().elements() != 1)
            return;
        const auto& mask_lens = where_ins->get_shape().lens();
        if(mask_lens.size() != 4)
            return;
        std::vector<std::size_t> lead_lens(mask_lens.begin(), mask_lens.end() - 2);
        if(lead_lens[0] * lead_lens[1] == 1)
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
        auto flat = m.insert_instruction(greater, make_op("reshape", {{"dims", {1}}}), seq_len);
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

void prepare_mlir::apply(module& m) const
{
    match::find_matches(m, find_reduce{}, find_leaky_relu{}, find_kv_cache_mask_seq_len{});
    match::find_matches(m, find_where{}, find_nonstandard_literal{});
    run_passes(m, {dead_code_elimination{}});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
