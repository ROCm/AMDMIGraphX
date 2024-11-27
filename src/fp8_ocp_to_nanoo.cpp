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
#include <migraphx/fp8_ocp_to_nanoo.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
    
struct match_fp8ocp_dq_convert_to_fp8nanoo
{
    /**
     * Match dequantizelinear instructions.
     * Bind the scale and zero_point inputs.
     */
    static auto dequantizelinear_op(const std::string& scale, const std::string& zp)
    {
        return match::name("dequantizelinear")(
            match::arg(1)(match::skip_broadcasts(match::is_constant().bind(scale))),
            match::arg(2)(match::skip_broadcasts(match::is_constant().bind(zp))));
    }

    auto matcher() const
    {
        return dequantizelinear_op("scale", "zp");
    }

    auto apply(module& m, const match::matcher_result& r) const
    {
        auto dq = r.result;
        auto x = dq->inputs().front();
        shape::type_t x_type = x->get_shape().type();
        if(x_type != shape::fp8e4m3fn_type)
        {
            return;
        }
        auto dq_scale = r.instructions["scale"];
        auto dq_zp = r.instructions["zp"];
    
        x = m.insert_instruction(dq, make_op("bit_cast", {{"target_type", shape::fp8e4m3fnuz_type}}), x);
        auto x_lens = x->get_shape().lens();

        // negative zero in fp8e4m3fn to zero in fp8e4m3fnuz
        // a == 0x80 ? 0x0 : a
        std::vector<fp8::fp8e4m3fnuz> bits_0x80 = {fp8::fp8e4m3fnuz(0x80, fp8::fp8e4m3fnuz::from_bits())};
        auto bits_0x80_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0x80);
        bits_0x80_lit = m.insert_instruction(dq, make_op("multibroadcast", {{"output_lens", x_lens}}), bits_0x80_lit);
        auto is_neg_zero = m.insert_instruction(dq, make_op("equal"), x, bits_0x80_lit);
        std::vector<fp8::fp8e4m3fnuz> bits_0x00 = {fp8::fp8e4m3fnuz(0x00, fp8::fp8e4m3fnuz::from_bits())};
        auto zero_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0x00);
        zero_lit = m.insert_instruction(dq, make_op("multibroadcast", {{"output_lens", x_lens}}), zero_lit);
        x = m.insert_instruction(dq, make_op("where"), is_neg_zero, zero_lit, x);

        // positive and negative NaN in fp8e4m3fn to NaN in fp8e4m3fnuz
        //(a & 0x7f) == 0x7f ? 0x80 : a
        std::vector<fp8::fp8e4m3fnuz> positive_nan_fp8ocp = {fp8::fp8e4m3fnuz(0x7f, fp8::fp8e4m3fnuz::from_bits())};
        auto nan_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, positive_nan_fp8ocp);
        nan_lit = m.insert_instruction(dq, make_op("multibroadcast", {{"output_lens", x_lens}}), nan_lit);
        auto cond = m.insert_instruction(dq, make_op("bitwise_and"), x, nan_lit);
        cond = m.insert_instruction(dq, make_op("equal"), cond, nan_lit);
        x = m.insert_instruction(dq, make_op("where"), cond, bits_0x80_lit, x);

        // adj_scale = 2 * scale
        auto two_lit = m.add_literal(literal{shape{dq_scale->get_shape().type()}, {2}});
        two_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", dq_scale->get_shape().lens()}}), two_lit);
        auto adj_dq_scale = m.insert_instruction(dq, make_op("mul"), dq_scale, two_lit);

        m.replace_instruction(dq, make_op("dequantizelinear"), x, adj_dq_scale, dq_zp);
        return;
    }
};

void fp8_ocp_to_nanoo::apply(module_pass_manager& mpm) const
{
    module_ref mm = &mpm.get_module();
    match::find_matches(*mm, match_fp8ocp_dq_convert_to_fp8nanoo{});
    mpm.run_pass(migraphx::dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
