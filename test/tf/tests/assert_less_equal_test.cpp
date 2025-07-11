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
 *
 */

#include <tf_test.hpp>

// Skip this test for libstdc++ debug for now since it exposes a bug in protobuf
#ifndef _GLIBCXX_DEBUG
TEST_CASE(assert_less_equal_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {2, 3}};
    auto l0 = mm->add_parameter("0", s0);
    auto l1 = mm->add_parameter("1", s0);
    migraphx::literal l{migraphx::shape{migraphx::shape::int32_type, {2}}, {0, 1}};
    auto l2 = mm->add_literal(l);
    mm->add_instruction(migraphx::make_op("add"), l0, l1);
    auto l3 = mm->add_instruction(migraphx::make_op("identity"), l0, l1);
    mm->add_instruction(migraphx::make_op("identity"), l3, l2);
    auto prog = optimize_tf("assert_less_equal_test.pb", false);

    EXPECT(p == prog);
}
#endif
