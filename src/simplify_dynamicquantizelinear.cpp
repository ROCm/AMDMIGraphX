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

#include <migraphx/simplify_dynamicquantizelinear.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

/*
 *  Dynamicquantizelinear by default adds uint8_t typed zero point into a quantize linear
 *  which needs to converted to int8 in order to avoid uint8 x int8 operations or uint8 operations
 *  from occuring on the backend as this isn't supported by MLIR nor how we simplify our quantizable
 *  ops.
 */
struct match_find_dynamicquantizelinear_convert_int8_zp
{
    auto matcher() const
    {
        return match::any_arg(0, 1)(
            match::name(get_quantizable_op_names())(
                match::any_arg(0, 1)(skip_broadcast_squeeze(
                    match::name("quantizelinear")(
                        match::arg(0)(skip_broadcasts(match::any())),
                        match::arg(2)(skip_broadcasts(
                            match::name("convert")(
                                match::has_type(migraphx::shape::uint8_type),
                                match::arg(0)(
                                    match::name("nearbyint")(
                                        match::arg(0)(match::name("clip").bind("saturate")))
                                        .bind("round")))
                                .bind("convert"))))
                        .bind("quant_lin"))))
                .bind("target"));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto target_op = r.instructions["target"];
        /* Need to modify the uint8 min/max range as well as final convert to convert to int8 */
        auto convert_op = r.instructions["convert"];
        // Ops to get q_min/q_max quickly
        auto round_op    = r.instructions["round"];
        auto quant_op    = r.instructions["quant_lin"];
        auto saturate_op = r.instructions["saturate"];
        auto q_min       = saturate_op->inputs().at(1);
        auto q_max       = saturate_op->inputs().at(2);

        // get new desired range defined by int8_t
        const auto x_min = std::numeric_limits<int8_t>::min();
        const auto x_max = std::numeric_limits<int8_t>::max();

        // Replace min/max of uint8 with min/max of int8 - q_range is identical so doesn't need to
        // be modified. Need to replace other ops which also take uint8 values first.
        auto x_type     = q_min->get_shape().type();
        auto q_min_int8 = m.add_literal(
            migraphx::literal{migraphx::shape{x_type, q_min->get_shape().lens()}, {x_min}});
        auto q_max_int8 = m.add_literal(
            migraphx::literal{migraphx::shape{x_type, q_max->get_shape().lens()}, {x_max}});

        m.replace_instruction(q_min, q_min_int8);
        m.replace_instruction(q_max, q_max_int8);

        auto new_conv = m.insert_instruction(
            convert_op,
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            round_op);

        // Convert inputs to the target op to ensure we're all int8 as part of our splice
        // This will be optimized out as part of simplify_reshapes if convert is redundant
        auto inputs = target_op->inputs();
        std::vector<instruction_ref> converted_inputs;
        for(auto& in : inputs)
        {
            auto item = in;
            if(in->get_shape().type() == migraphx::shape::uint8_type)
            {
                item = m.insert_instruction(
                    target_op,
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
                    in);
            }
            converted_inputs.push_back(item);
        }

        auto new_target_op =
            m.insert_instruction(target_op, target_op->get_operator(), converted_inputs);

        auto data  = quant_op->inputs().at(0);
        auto scale = quant_op->inputs().at(1);

        m.replace_instruction(target_op, new_target_op);
        m.remove_instruction(target_op);

        auto new_quant = m.insert_instruction(
            quant_op,
            migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::int8_type}}),
            {data, scale, new_conv});

        m.replace_instruction(convert_op, new_conv);
        m.remove_instruction(convert_op);
        m.move_instruction(new_quant, quant_op);
    }
};

void simplify_dynamicquantizelinear::apply(module& m) const
{
    match::find_matches(m, match_find_dynamicquantizelinear_convert_int8_zp{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
