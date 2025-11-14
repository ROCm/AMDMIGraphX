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

TEST_CASE(addn_op_builder_test)
{
    migraphx::module mm;
    auto arg0 = mm.add_parameter("0", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg1 = mm.add_parameter("1", {migraphx::shape::float_type, {2, 4, 5}});
    auto arg2 = mm.add_parameter("2", {migraphx::shape::float_type, {2, 4, 5}});
    auto add1 = mm.add_instruction(migraphx::make_op("add"), {arg0, arg1});
    mm.add_instruction(migraphx::make_op("add"), {add1, arg2});

    EXPECT(mm == make_op_module("addn", migraphx::value("", {}, false), mm.get_parameters()));
}
