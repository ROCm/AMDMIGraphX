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

#include <op_builder_test_utils.hpp>

TEST_CASE(binary_no_op_name_specified_test)
{
    migraphx::module mm;

    EXPECT(test::throws<migraphx::exception>(
            [&] { make_op_module("binary", {}, mm.get_parameters()); },
            "Binary op missing op_name attribute"));    
}

TEST_CASE(binary_dynamic_input_shape_test)
{
    migraphx::module mm;

    migraphx::shape::dynamic_dimension dd{1, 4};
    std::vector<migraphx::shape::dynamic_dimension> dyn_dims{dd, dd};

    mm.add_parameter("a", {migraphx::shape::float_type, dyn_dims});
    mm.add_parameter("b", {migraphx::shape::float_type, dyn_dims});

    EXPECT(test::throws<migraphx::exception>(
            [&] { make_op_module("binary", 
                {{"op_name", "DONTCARE"}, {"is_broadcasted", true}, {"broadcasted", 1}}, 
                mm.get_parameters()); },
            "Binary op broadcast attribute not supported for dynamic input shapes"));    
}

TEST_CASE(binary_not_broadcasted_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {2, 4}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4}});

    add_common_op(mm, migraphx::make_op("add"), {a_arg, b_arg});

        EXPECT(mm == make_op_module("binary",
                                {{"op_name", "add"}},
                                mm.get_parameters()));
}

TEST_CASE(binary_zero_broadcasted_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {2, 4}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4}});

    mm.add_instruction(migraphx::make_op("add"), {a_arg, b_arg});

    EXPECT(mm == make_op_module("binary",
                            {{"op_name", "add"}, {"is_broadcasted", true}, {"broadcasted", 0}},
                            mm.get_parameters()));
}

TEST_CASE(binary_non_zero_broadcasted_test)
{
    migraphx::module mm;
    auto a_arg = mm.add_parameter("a", {migraphx::shape::float_type, {2, 4}});
    auto b_arg = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4}});

    auto l = mm.add_instruction(migraphx::make_op("broadcast",{{"axis", 0}, {"out_lens", {2, 4}}}), b_arg);
    mm.add_instruction(migraphx::make_op("add"), {a_arg, l});

    EXPECT(mm == make_op_module("binary",
                            {{"op_name", "add"}, {"is_broadcasted", true}, {"broadcasted", 1}, {"axis", 0}},
                            mm.get_parameters()));
}
