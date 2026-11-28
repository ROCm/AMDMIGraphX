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
 */
#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <quantize_helpers.hpp>

template <std::size_t Rank, std::size_t Channels>
struct test_qdq_conv_expanded : verify_program<test_qdq_conv_expanded<Rank, Channels>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* m = p.get_main_module();
        // Singleton spatial dimensions avoid MLIR's broadcasted output-scale reshape bug.
        std::vector<std::size_t> input_lens(Rank, 1);
        input_lens[1] = 2;
        std::vector<std::size_t> weight_lens(Rank, 1);
        weight_lens[0] = Channels;
        weight_lens[1] = 2;
        migraphx::shape input_shape{migraphx::shape::int8_type, input_lens};
        migraphx::shape weight_shape{migraphx::shape::int8_type, weight_lens};
        auto input  = m->add_parameter("input", input_shape);
        auto weight = m->add_parameter("weight", weight_shape);
        std::vector<std::size_t> qlens(Rank, 1);
        qlens[0] = Channels;
        std::vector<float> scales(Channels, 0.25f);
        std::vector<std::int8_t> zeros(Channels, 0);
        auto scale =
            m->add_literal(migraphx::literal{{migraphx::shape::float_type, qlens}, scales});
        auto zero = m->add_literal(migraphx::literal{{migraphx::shape::int8_type, qlens}, zeros});
        auto si   = m->add_literal(0.5f);
        auto zi   = m->add_literal(std::int8_t{0});
        auto di   = add_quantize_op(*m, "dequantizelinear", input, si, zi);
        auto sb   = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", weight_lens}}), scale);
        auto zb = m->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", weight_lens}}), zero);
        auto dw = m->add_instruction(migraphx::make_op("dequantizelinear"), weight, sb, zb);
        std::vector<int64_t> padding(Rank - 2, 0);
        std::vector<int64_t> stride(Rank - 2, 1);
        auto conv = m->add_instruction(
            migraphx::make_op("convolution",
                              {{"padding", padding}, {"stride", stride}, {"dilation", stride}}),
            di,
            dw);
        m->add_return({conv});
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_qdq_conv_expanded<3, 1>;
template struct test_qdq_conv_expanded<3, 2>;
template struct test_qdq_conv_expanded<4, 1>;
template struct test_qdq_conv_expanded<4, 2>;
template struct test_qdq_conv_expanded<5, 1>;
template struct test_qdq_conv_expanded<5, 2>;
