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
namespace {

using fp8::fp8e4m3fnuz;

template <class... Ms>
auto skip_post_dq_ops(Ms... ms)
{
    return match::skip(match::name(
        "broadcast", "multibroadcast", "contiguous", "transpose", "reshape", "convert"))(ms...);
}

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

struct match_fp8ocp_convert_to_fp8nanoo
{
    // almost the same as matcher from simplify_qdq
    // difference is that broadcasts are not skipped
    static auto dequantizelinear_op(const std::string& scale, const std::string& zp)
    {
        return match::name("dequantizelinear")(match::arg(1)(match::is_constant().bind(scale)),
                                               match::arg(2)(match::is_constant().bind(zp)));
    }
    auto matcher() const
    {
        auto dq1 =
            match::arg(0)(skip_post_dq_ops(dequantizelinear_op("scale1", "zp1").bind("dq1")));
        auto dq2 =
            match::arg(1)(skip_post_dq_ops(dequantizelinear_op("scale2", "zp2").bind("dq2")));
        return match::name(get_quantizable_op_names())(dq1, dq2);
    }

    auto bit_cast_and_handle_specials(module& m,
                                      const instruction_ref dq,
                                      const instruction_ref x,
                                      const instruction_ref bits_0x80_lit,
                                      const instruction_ref bits_0x7f_lit,
                                      const instruction_ref bits_0xff_lit,
                                      const instruction_ref bits_0x00_lit) const
    {
        auto x_lens = x->get_shape().lens();
        auto cast_input = m.insert_instruction(
            dq, make_op("bit_cast", {{"target_type", shape::fp8e4m3fnuz_type}}), x);
        auto mb_bits_0x80_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x80_lit);
        auto mb_bits_0x7f_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x7f_lit);
        auto mb_bits_0xff_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0xff_lit);
        auto mb_zero_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", x_lens}}), bits_0x00_lit);
        // negative zero in fp8e4m3fn to zero in fp8e4m3fnuz
        // a == 0x80 ? 0x0 : a
        auto is_neg_zero = m.insert_instruction(dq, make_op("equal"), cast_input, mb_bits_0x80_lit);
        auto ret = m.insert_instruction(dq, make_op("where"), is_neg_zero, mb_zero_lit, cast_input);

        // positive and negative NaN in fp8e4m3fn to NaN in fp8e4m3fnuz
        // (a == 0x7f or a == 0xff) ? 0x80 : a
        auto eq_0x7f = m.insert_instruction(dq, make_op("equal"), ret, mb_bits_0x7f_lit);

        auto eq_0xff = m.insert_instruction(dq, make_op("equal"), ret, mb_bits_0xff_lit);

        auto cond = m.insert_instruction(dq, make_op("logical_or"), eq_0x7f, eq_0xff);
        ret       = m.insert_instruction(dq, make_op("where"), cond, mb_bits_0x80_lit, ret);
        return ret;
    }

    auto cast_to_nanoo(module& m,
                       const instruction_ref dq,
                       const instruction_ref input,
                       const instruction_ref dq_scale,
                       const instruction_ref dq_zp) const
    {
        auto x                             = input;
        std::vector<fp8e4m3fnuz> bits_0x80 = {fp8e4m3fnuz(0x80, fp8e4m3fnuz::from_bits())};
        auto bits_0x80_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0x80);

        std::vector<fp8e4m3fnuz> bits_0x7f = {fp8e4m3fnuz(0x7f, fp8e4m3fnuz::from_bits())};
        auto bits_0x7f_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0x7f);

        std::vector<fp8e4m3fnuz> bits_0xff = {fp8e4m3fnuz(0xff, fp8e4m3fnuz::from_bits())};
        auto bits_0xff_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0xff);

        std::vector<fp8e4m3fnuz> bits_0x00 = {fp8e4m3fnuz(0x00, fp8e4m3fnuz::from_bits())};
        auto bits_0x00_lit = m.add_literal(shape{shape::fp8e4m3fnuz_type, {1}}, bits_0x00);

        x = bit_cast_and_handle_specials(
            m, dq, x, bits_0x80_lit, bits_0x7f_lit, bits_0xff_lit, bits_0x00_lit);
        auto adj_zp = bit_cast_and_handle_specials(
            m, dq, dq_zp, bits_0x80_lit, bits_0x7f_lit, bits_0xff_lit, bits_0x00_lit);

        // adj_scale = 2 * scale
        auto two_lit = m.add_literal(literal{shape{dq_scale->get_shape().type()}, {2}});
        two_lit = m.insert_instruction(
            dq, make_op("multibroadcast", {{"out_lens", dq_scale->get_shape().lens()}}), two_lit);
        auto adj_dq_scale = m.insert_instruction(dq, make_op("mul"), dq_scale, two_lit);

        m.replace_instruction(dq, make_op("dequantizelinear"), x, adj_dq_scale, adj_zp);
    }

    auto apply(module& m, const match::matcher_result& r) const
    {
        auto dq1    = r.instructions["dq1"];
        auto dq2    = r.instructions["dq2"];
        auto scale1 = r.instructions["scale1"];
        auto scale2 = r.instructions["scale2"];
        auto zp1    = r.instructions["zp1"];
        auto zp2    = r.instructions["zp2"];

        std::set<migraphx::shape::type_t> supported_types = {migraphx::shape::fp8e4m3fn_type};
        if(not contains(supported_types, dq1->inputs().front()->get_shape().type()) or
           not contains(supported_types, dq2->inputs().front()->get_shape().type()))
            return;

        cast_to_nanoo(m, dq1, dq1->inputs().front(), scale1, zp1);
        cast_to_nanoo(m, dq2, dq2->inputs().front(), scale2, zp2);
        return;
    }
};

} // namespace

void fp8_ocp_to_nanoo::apply(module_pass_manager& mpm) const
{
    module_ref mm = &mpm.get_module();
    match::find_matches(*mm, match_fp8ocp_convert_to_fp8nanoo{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
