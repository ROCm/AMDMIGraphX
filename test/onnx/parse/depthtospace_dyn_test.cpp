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

#include <onnx_test.hpp>

TEST_CASE(depthtospace_dyn_dcr_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x =
        mm->add_parameter("x", {migraphx::shape::float_type, {{2, 4}, {16, 16}, {5, 10}, {5, 10}}});
    auto blocksize_literal = mm->add_literal(static_cast<int64_t>(2));
    auto n = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), x);
    auto h = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 3}}), x);
    auto w = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 3}, {"end", 4}}), x);
    auto c_div = mm->add_literal(static_cast<int64_t>(4));

    auto new_shape1 = mm->add_instruction(
        migraphx::make_op("concat"), n, blocksize_literal, blocksize_literal, c_div, h, w);

    auto dyn_dims1        = migraphx::shape{migraphx::shape::float_type,
                                            {{2, 4}, {2, 2}, {2, 2}, {4, 4}, {5, 10}, {5, 10}}};
    auto new_shape_alloc1 = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", to_value(dyn_dims1)}}), new_shape1);

    auto reshape1 = mm->add_instruction(migraphx::make_op("reshape"), x, new_shape_alloc1);

    std::vector<int64_t> perm = {0, 3, 4, 1, 5, 2};
    auto transpose =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), reshape1);
    auto h_blocksize = mm->add_instruction(migraphx::make_op("mul"), h, blocksize_literal);
    auto w_blocksize = mm->add_instruction(migraphx::make_op("mul"), w, blocksize_literal);
    auto new_shape2 =
        mm->add_instruction(migraphx::make_op("concat"), n, c_div, h_blocksize, w_blocksize);

    auto dyn_dims2 =
        migraphx::shape{migraphx::shape::float_type, {{2, 4}, {4, 4}, {10, 20}, {10, 20}}};

    auto new_shape_alloc2 = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", to_value(dyn_dims2)}}), new_shape2);
    mm->add_instruction(migraphx::make_op("reshape"), transpose, new_shape_alloc2);

    auto prog = optimize_onnx(
        "depthtospace_dyn_dcr_test.onnx", false, {{"x", {{2, 4}, {16, 16}, {5, 10}, {5, 10}}}});
    EXPECT(p == prog);
}

TEST_CASE(depthtospace_dyn_crd_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x =
        mm->add_parameter("x", {migraphx::shape::float_type, {{2, 4}, {16, 16}, {5, 10}, {5, 10}}});
    auto blocksize_literal = mm->add_literal(static_cast<int64_t>(2));
    auto n = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 0}, {"end", 1}}), x);
    auto h = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 2}, {"end", 3}}), x);
    auto w = mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 3}, {"end", 4}}), x);
    auto c_div = mm->add_literal(static_cast<int64_t>(4));

    auto new_shape1 = mm->add_instruction(
        migraphx::make_op("concat"), n, c_div, blocksize_literal, blocksize_literal, h, w);

    auto dyn_dims1        = migraphx::shape{migraphx::shape::float_type,
                                            {{2, 4}, {4, 4}, {2, 2}, {2, 2}, {5, 10}, {5, 10}}};
    auto new_shape_alloc1 = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", to_value(dyn_dims1)}}), new_shape1);

    auto reshape1 = mm->add_instruction(migraphx::make_op("reshape"), x, new_shape_alloc1);

    std::vector<int64_t> perm = {0, 1, 4, 2, 5, 3};
    auto transpose =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), reshape1);
    auto h_blocksize = mm->add_instruction(migraphx::make_op("mul"), h, blocksize_literal);
    auto w_blocksize = mm->add_instruction(migraphx::make_op("mul"), w, blocksize_literal);
    auto new_shape2 =
        mm->add_instruction(migraphx::make_op("concat"), n, c_div, h_blocksize, w_blocksize);

    auto dyn_dims2 =
        migraphx::shape{migraphx::shape::float_type, {{2, 4}, {4, 4}, {10, 20}, {10, 20}}};

    auto new_shape_alloc2 = mm->add_instruction(
        migraphx::make_op("allocate", {{"shape", to_value(dyn_dims2)}}), new_shape2);
    mm->add_instruction(migraphx::make_op("reshape"), transpose, new_shape_alloc2);

    auto prog = optimize_onnx(
        "depthtospace_dyn_crd_test.onnx", false, {{"x", {{2, 4}, {16, 16}, {5, 10}, {5, 10}}}});
    EXPECT(p == prog);
}
