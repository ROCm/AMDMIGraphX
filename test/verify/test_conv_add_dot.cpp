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
 */

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

template <migraphx::shape::type_t DType>
struct test_conv_add_dot : verify_program<test_conv_add_dot<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto input   = mm->add_parameter("x", migraphx::shape{DType, {4, 3, 3, 3}});
        auto weights = mm->add_parameter("w", migraphx::shape{DType, {4, 3, 3, 3}});
        auto bias_literal =
            migraphx::literal{migraphx::shape{DType, {4}}, {2.0f, 2.0f, 2.0f, 2.0f}};
        auto bias       = mm->add_literal(bias_literal);
        auto conv       = mm->add_instruction(migraphx::make_op("convolution"), input, weights);
        auto bcast_bias = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv, bcast_bias);

        // Create a literal for dot (matmul) with shape {4, 3, 3, 3}
        std::vector<std::size_t> bias_add_lens = bias_add->get_shape().lens();
        // The shape is {4, 3, 3, 3}, so we want a rhs shape {4, 3, 3, 3}
        migraphx::shape dot_rhs_shape{DType, bias_add_lens};
        std::vector<float> dot_rhs_data(dot_rhs_shape.elements(), 1.0f);
        auto dot_rhs = mm->add_literal(migraphx::literal{dot_rhs_shape, dot_rhs_data});

        // Matmul (dot) with same shape, so this is elementwise matmul
        auto dot = mm->add_instruction(migraphx::make_op("dot"), bias_add, dot_rhs);
        mm->add_return({dot});
        return p;
    }
    // std::string section() const { return "conv"; }
};

template struct test_conv_add_dot<migraphx::shape::half_type>;
template struct test_conv_add_dot<migraphx::shape::bf16_type>;
