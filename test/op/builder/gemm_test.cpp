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

#include <op_builder_test_utils.hpp>
#include <migraphx/instruction.hpp>

TEST_CASE(gemm_invalid_input_dim_op_builder_test)
{
    migraphx::module mm;
    mm.add_parameter("a", {migraphx::shape::float_type, {3}});
    mm.add_parameter("b", {migraphx::shape::float_type, {3, 3, 3}});

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("gemm", mm.get_parameters()); },
        "gemm op_builder: A and B should be rank 2, A is rank 1, B is rank 3"));
}

TEST_CASE(gemm_normal_path_op_builder_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {3, 3}});

    a_arg = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), a_arg);
    b_arg = mm.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), b_arg);
    mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    EXPECT(mm == make_op_module("gemm",
                                {{"alpha", 1.0f}, {"transA", true}, {"transB", true}},
                                mm.get_parameters()));
}

TEST_CASE(gemm_alpha_not_one_op_builder_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {3, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {3, 3}});

    const float alpha = 1.1f;

    auto alpha_literal = mm.add_literal(alpha);
    a_arg              = add_common_op(mm, migraphx::make_op("mul"), {alpha_literal, a_arg});

    mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    EXPECT(mm == make_op_module("gemm",
                                {{"alpha", alpha}, {"transA", false}, {"transB", false}},
                                mm.get_parameters()));
}

TEST_CASE(gemm_alpha_not_one_type_mismatch_op_builder_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::fp8e4m3fnuz_type, {3, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::fp8e4m3fnuz_type, {3, 3}});

    const float alpha   = 1.1f;
    const auto dot_type = a_arg->get_shape().type();

    auto alpha_literal = mm.add_literal(alpha);
    a_arg              = add_common_op(mm, migraphx::make_op("mul"), {alpha_literal, a_arg});
    a_arg = mm.add_instruction(migraphx::make_op("convert", {{"target_type", dot_type}}), a_arg);

    mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    EXPECT(mm == make_op_module("gemm",
                                {{"alpha", alpha}, {"transA", false}, {"transB", false}},
                                mm.get_parameters()));
}

TEST_CASE(gemm_3_params_not_dynamic_op_builder_test)
{
    migraphx::module mm;

    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {1, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4}});
    auto c_arg = mm.add_parameter("c", {migraphx::shape::float_type, {1, 1}});

    auto dot_ins = mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    c_arg = mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4}}}), c_arg);
    mm.add_instruction(migraphx::make_op("add"), dot_ins, c_arg);

    EXPECT(mm ==
           make_op_module("gemm",
                          {{"alpha", 1.0f}, {"transA", false}, {"transB", false}, {"beta", 1.0f}},
                          mm.get_parameters()));
}

TEST_CASE(gemm_3_params_dynamic_op_builder_test)
{
    migraphx::module mm;

    migraphx::shape::dynamic_dimension dd{1, 4};
    std::vector<migraphx::shape::dynamic_dimension> dyn_dims{dd, dd};

    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, dyn_dims});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, dyn_dims});
    auto c_arg = mm.add_parameter("c", {migraphx::shape::float_type, {1, 1}});

    auto dot_ins = mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    c_arg = mm.add_instruction(migraphx::make_op("multibroadcast"), c_arg, dot_ins);
    mm.add_instruction(migraphx::make_op("add"), dot_ins, c_arg);

    EXPECT(mm ==
           make_op_module("gemm",
                          {{"alpha", 1.0f}, {"transA", false}, {"transB", false}, {"beta", 1.0f}},
                          mm.get_parameters()));
}

TEST_CASE(gemm_3_params_beta_not_one_op_builder_test)
{
    migraphx::module mm;

    const float beta = 1.1f;

    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {1, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {3, 4}});
    auto c_arg = mm.add_parameter("c", {migraphx::shape::float_type, {1, 4}});

    auto dot_ins = mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    auto beta_literal = mm.add_literal(beta);
    c_arg             = add_common_op(mm, migraphx::make_op("mul"), {c_arg, beta_literal});
    mm.add_instruction(migraphx::make_op("add"), dot_ins, c_arg);

    EXPECT(mm ==
           make_op_module("gemm",
                          {{"alpha", 1.0f}, {"transA", false}, {"transB", false}, {"beta", beta}},
                          mm.get_parameters()));
}

TEST_CASE(gemm_3_params_beta_not_one_type_mismatch_op_builder_test)
{
    migraphx::module mm;

    const float beta = 0.8f;

    auto a_arg = mm.add_parameter("a", {migraphx::shape::bf16_type, {1, 3}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::bf16_type, {3, 4}});
    auto c_arg = mm.add_parameter("c", {migraphx::shape::bf16_type, {1, 4}});

    auto dot_ins = mm.add_instruction(migraphx::make_op("dot"), a_arg, b_arg);

    auto beta_literal = mm.add_literal(beta);
    c_arg             = add_common_op(mm, migraphx::make_op("mul"), {c_arg, beta_literal});
    c_arg             = mm.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bf16_type}}), c_arg);
    mm.add_instruction(migraphx::make_op("add"), dot_ins, c_arg);

    EXPECT(mm ==
           make_op_module("gemm",
                          {{"alpha", 1.0f}, {"transA", false}, {"transB", false}, {"beta", beta}},
                          mm.get_parameters()));
}
