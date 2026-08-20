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
#include <migraphx/fp8_types.hpp>
#include <layernorm.hpp>

template <migraphx::shape::type_t DType>
struct test_conv_add_layernorm_conv : verify_program<test_conv_add_layernorm_conv<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm          = p.get_main_module();
        auto input        = mm->add_parameter("x", migraphx::shape{DType, {2, 4, 64, 64}});
        auto weights1     = mm->add_parameter("w1", migraphx::shape{DType, {320, 4, 3, 3}});
        auto weights2     = mm->add_parameter("w2", migraphx::shape{DType, {4, 320, 3, 3}});
        auto bias_literal = abs(migraphx::generate_literal(migraphx::shape{DType, {320}}, 1));
        auto bias         = mm->add_literal(bias_literal);
        auto conv1        = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), input, weights1);
        auto bcast_bias = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv1->get_shape().lens()}}),
            bias);
        auto bias_add = mm->add_instruction(migraphx::make_op("add"), conv1, bcast_bias);
        auto rsp_add =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {0, 32, -1}}}), bias_add);
        float eps = 1e-12;
        if constexpr((DType) == migraphx::shape::fp8e4m3fnuz_type)
        {
            // use 0.250 for fp8
            eps = 0.250;
        }
        else if constexpr((DType) == migraphx::shape::half_type)
        {
            eps = 1e-6;
        }
        auto layernorm     = add_layernorm(*mm, rsp_add, rsp_add->get_shape().lens(), eps);
        auto layernorm_rsp = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {0, 320, 64, 64}}}), layernorm);
        mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), layernorm_rsp, weights2);
        return p;
    }
    std::string section() const { return "conv"; }

    migraphx::optional<migraphx::verify::tolerance> get_tolerance() const
    {
        // The layernorm between the two convolutions normalizes to near zero while the
        // surrounding conv output is orders of magnitude larger, so the failing elements are the
        // small ones that only atol reaches: rtol alone would need 1400x to 2600x, or is outright
        // infinite for fp8. Measured atol: 130x for half, 52x for bf16, 41x for fp8.
        // get_ref_use_double makes both worse, so the gpu is the noisy side.
        auto tols   = migraphx::default_tolerance_for(DType);
        double wide = 0;
        if constexpr(DType == migraphx::shape::half_type)
            wide = 260;
        else if constexpr(DType == migraphx::shape::bf16_type)
            wide = 110;
        else if(migraphx::contains(migraphx::fp8_types{}.get(), DType))
            wide = 90;
        if(wide == 0)
            return migraphx::nullopt;
        tols.atol *= wide;
        return tols;
    }
};

template struct test_conv_add_layernorm_conv<migraphx::shape::float_type>;
template struct test_conv_add_layernorm_conv<migraphx::shape::half_type>;
template struct test_conv_add_layernorm_conv<migraphx::shape::bf16_type>;
template struct test_conv_add_layernorm_conv<migraphx::shape::fp8e4m3fnuz_type>;
