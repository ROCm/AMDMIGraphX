/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <op_builder_test_utils.hpp>

TEST_CASE(gemm_invalid_input_dim_op_builder_test)
{
    migraphx::module mm;
    mm.add_parameter("a", {migraphx::shape::float_type, {3}});
    mm.add_parameter("b", {migraphx::shape::float_type, {3,3,3}});
    
    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("gemm", {}, mm.get_parameters()); },
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

TEST_CASE(gemm_alpha_one_op_builder_test)
{
    
}

TEST_CASE(gemm_alpha_one_not_dot_type_op_builder_test)
{
    
}

/*
TEST_CASE(gemm__op_builder_test)
{
    
}

TEST_CASE(gemm__op_builder_test)
{
    
}

TEST_CASE(gemm__op_builder_test)
{
    
}

TEST_CASE(gemm__op_builder_test)
{
    
}
*/
