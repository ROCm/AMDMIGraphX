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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/rewrite_rmsnorm.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/match/rmsnorm.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct rewrite_rmsnorm_dtype
{
    auto matcher() const { return match::rmsnorm(); }

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

        auto reduce_op = r.instructions["reduce_op"];
        auto converted_reduce_op =
            m.insert_instruction(reduce_op, reduce_op->get_operator(), converted_pow);

        // NOTE: Eps (and add) will be optimized out, revisit after issue #2659 is resolved
        auto variance = converted_reduce_op;
        if(r.instructions.find("add") != r.instructions.end())
        {
            auto add = r.instructions["add"];
            std::vector<instruction_ref> add_inputs;
            add_inputs.emplace_back(converted_reduce_op);
            add_inputs.emplace_back(
                m.insert_instruction(add,
                                     make_op("convert", {{"target_type", shape::float_type}}),
                                     add->inputs().at(1)));
            variance = m.insert_instruction(add, add->get_operator(), add_inputs);
        }

        auto sqrt           = r.instructions["sqrt"];
        auto converted_sqrt = m.insert_instruction(sqrt, sqrt->get_operator(), variance);

        instruction_ref prev_input;
        instruction_ref converted_input;

        if(r.instructions.find("mbcast") != r.instructions.end())
        {
            // mul -- mbcast -- rsqrt path
            prev_input = r.instructions["mbcast"];
            converted_input =
                m.insert_instruction(prev_input, prev_input->get_operator(), converted_sqrt);
        }
        else
        {
            // 1/div -- sqrt path
            prev_input = r.instructions["div"];
            std::vector<instruction_ref> prev_input_inputs;
            prev_input_inputs.emplace_back(
                m.insert_instruction(prev_input,
                                     make_op("convert", {{"target_type", shape::float_type}}),
                                     prev_input->inputs().at(0)));
            prev_input_inputs.emplace_back(converted_sqrt);
            converted_input =
                m.insert_instruction(prev_input, prev_input->get_operator(), prev_input_inputs);
        }

        auto y = prev_input->outputs().at(0);
        std::vector<instruction_ref> y_inputs;
        if(y->inputs().size() > 1)
        {
            auto other_input =
                y->inputs().at(0) == prev_input ? y->inputs().at(1) : y->inputs().at(0);
            y_inputs.emplace_back(other_input);
        }
        y_inputs.emplace_back(m.insert_instruction(
            y, make_op("convert", {{"target_type", shape::half_type}}), converted_input));
        auto converted_y = m.insert_instruction(y, y->get_operator(), y_inputs);

        m.replace_instruction(y, converted_y);
    }
};

void rewrite_rmsnorm::apply(module& m) const
{
    match::find_matches(m, rewrite_rmsnorm_dtype{});
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
