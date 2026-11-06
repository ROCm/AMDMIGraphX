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
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct bias_uint8 : op_builder<bias_uint8>
{
    bool has_bias = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.has_bias, "has_bias"));
    }

    // Convert to half prior to a shift to ensure we preserve accuracy here then
    // convert back to int8
    instruction_ref add_int8_shift(module& m,
                                   instruction_ref ins,
                                   const instruction_ref& offset_op,
                                   instruction_ref& unshifted_input) const
    {
        auto unshifted_input_half = m.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            unshifted_input);

        auto input_shifted_half = insert_common_op(m, ins, "add", unshifted_input_half, offset_op);

        return m.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_half);
    }

    void shift_input_and_bias(module& m,
                              instruction_ref ins,
                              instruction_ref& input,
                              instruction_ref& input_bias) const
    {
        auto offset_op =
            m.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-128}});

        input = add_int8_shift(m, ins, offset_op, input);
        if(has_bias)
        {
            input_bias = add_int8_shift(m, ins, offset_op, input_bias);
        }
        else
        {
            input_bias = input;
        }
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto arg      = args[0];
        auto bias_arg = args[1];

        // always convert uint8 to int8 to avoid rollover
        if(arg->get_shape().type() == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(m, ins, arg, bias_arg);
        }

        // subtract bias from result after conversion
        if(has_bias)
        {
            bias_arg = insert_common_op(m, ins, "sub", arg, bias_arg);
        }

        return {arg, bias_arg};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
