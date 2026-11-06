/* The MIT License (MIT)
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
 */

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct gelu_quick : op_builder<gelu_quick>
{
    float alpha = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x         = args[0];
        auto x_type    = x->get_shape().type();
        auto alpha_lit = m.add_literal(migraphx::literal{migraphx::shape{x_type}, {alpha}});
        auto mul_alpha = insert_common_op(m, ins, make_op("mul"), {alpha_lit, x});
        auto sigmoid   = m.insert_instruction(ins, migraphx::make_op("sigmoid"), mul_alpha);
        return {insert_common_op(m, ins, make_op("mul"), {x, sigmoid})};
    }
};

struct gelu_erf : op_builder<gelu_erf>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x        = args[0];
        auto x_type   = x->get_shape().type();
        auto half     = m.add_literal({x_type, {0.5f}});
        auto one      = m.add_literal({x_type, {1.0f}});
        auto sqrt2    = m.add_literal({x_type, {static_cast<float>(M_SQRT2)}});
        auto mul_half = insert_common_op(m, ins, "mul", x, half);
        auto div      = insert_common_op(m, ins, "div", x, sqrt2);
        auto erf      = m.insert_instruction(ins, migraphx::make_op("erf"), div);
        auto add_one  = insert_common_op(m, ins, "add", erf, one);
        return {insert_common_op(m, ins, "mul", mul_half, add_one)};
    }
};

struct gelu_tanh : op_builder<gelu_tanh>
{
    bool fast = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.fast, "fast"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x      = args[0];
        auto x_type = x->get_shape().type();

        auto fit_const_val  = fast ? 0.035677 : 0.044715;
        auto fit_const      = m.add_literal({x_type, {fit_const_val}});
        auto sqrt_2_rpi_val = fast ? 0.797885 : sqrt(M_2_PI);
        auto sqrt_2_rpi     = m.add_literal({x_type, {sqrt_2_rpi_val}});
        auto one            = m.add_literal({x_type, {1.0f}});
        auto half           = m.add_literal({x_type, {0.5f}});
        auto three          = m.add_literal({x_type, {3.0f}});

        // [0.044715|0.035677] * x^3
        auto pow0 = insert_common_op(m, ins, "pow", x, three);
        auto mul0 = insert_common_op(m, ins, "mul", pow0, fit_const);
        instruction_ref tanh_in;
        if(fast)
        {
            // approx = 0.797885 * x + 0.035677 * x^3
            auto mul1 = insert_common_op(m, ins, "mul", sqrt_2_rpi, x);
            tanh_in   = insert_common_op(m, ins, "add", mul0, mul1);
        }
        else
        {
            // approx = sqrt(2/pi) * (x + 0.044715 * x^3
            auto add0 = insert_common_op(m, ins, "add", mul0, x);
            tanh_in   = insert_common_op(m, ins, "mul", add0, sqrt_2_rpi);
        }

        // 0.5 * x * (1 + Tanh(approx))
        auto tanh0 = insert_common_op(m, ins, "tanh", tanh_in);
        auto add1  = insert_common_op(m, ins, "add", tanh0, one);
        auto mul2  = insert_common_op(m, ins, "mul", x, half);
        return {insert_common_op(m, ins, "mul", add1, mul2)};
    }
};

struct gelu_split : op_builder<gelu_split>
{
    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x               = args[0];
        size_t last_dim_size = x->get_shape().lens().back();
        if(last_dim_size < 2 or last_dim_size % 2 != 0)
            MIGRAPHX_THROW(
                "gelu_split op_builder: BiasSplitGelu must have even last dimension which is >= 2");

        auto split_left = m.add_instruction(
            migraphx::make_op("slice",
                              {{"axes", {-1}}, {"starts", {0}}, {"ends", {last_dim_size / 2}}}),
            x);
        auto split_right = m.add_instruction(
            migraphx::make_op(
                "slice",
                {{"axes", {-1}}, {"starts", {last_dim_size / 2}}, {"ends", {last_dim_size}}}),
            x);

        auto gelu_erf = op::builder::add("gelu_erf", m, {split_right}).at(0);
        return {insert_common_op(m, ins, "mul", split_left, gelu_erf)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
