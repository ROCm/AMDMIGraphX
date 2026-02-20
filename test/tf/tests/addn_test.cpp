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

TEST_CASE(addn_test)
{
    migraphx::program p;

    auto* mm  = p.get_main_module();
    auto l0   = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto l1   = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto l2   = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    auto add1 = mm->add_instruction(migraphx::make_op("add"), l0, l1);
    mm->add_instruction(migraphx::make_op("add"), add1, l2);
    auto prog = optimize_tf("addn_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(addn_single_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_tf("addn_single_test.pb", false);

    EXPECT(p == prog);
}

TEST_CASE(addn_with_10_elements_test)
{
    migraphx::program p;

    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l1  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l2  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l3  = mm->add_parameter("3", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l4  = mm->add_parameter("4", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l5  = mm->add_parameter("5", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l6  = mm->add_parameter("6", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l7  = mm->add_parameter("7", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l8  = mm->add_parameter("8", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto l9  = mm->add_parameter("9", migraphx::shape{migraphx::shape::float_type, {1, 1648}});
    auto us0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto us1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l1);
    auto us2 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l2);
    auto us3 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l3);
    auto us4 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l4);
    auto us5 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l5);
    auto us6 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l6);
    auto us7 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l7);
    auto us8 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l8);
    auto us9 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l9);
    auto concatenated = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}),
                                            us0,
                                            us1,
                                            us2,
                                            us3,
                                            us4,
                                            us5,
                                            us6,
                                            us7,
                                            us8,
                                            us9);
    auto reduced =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), concatenated);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), reduced);
    auto prog = optimize_tf("addn_with_many_elements_test.pb", false);

    EXPECT(p == prog);
}
