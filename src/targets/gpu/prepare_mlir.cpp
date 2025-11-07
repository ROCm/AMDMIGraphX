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
        auto alpha    = m.add_literal(literal{{x_ins->get_shape().type(), {1}}, {alpha_f}});
        auto zero     = m.add_literal(literal{{x_ins->get_shape().type(), {1}}, {0.0}});

        auto greater   = insert_common_op(m, ins, make_op("greater"), {x_ins, zero});
        auto mul_alpha = insert_common_op(m, ins, make_op("mul"), {x_ins, alpha});

        m.replace_instruction(ins, make_op("where"), {greater, x_ins, mul_alpha});
    }
};

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

} // namespace

void prepare_mlir::apply(module& m) const
{
    match::find_matches(m, find_reduce{}, find_leaky_relu{});
    match::find_matches(m, find_where{});
    run_passes(m, {dead_code_elimination{}});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
