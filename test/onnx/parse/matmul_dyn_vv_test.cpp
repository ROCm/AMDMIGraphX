/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/apply_alpha_beta.hpp>

TEST_CASE(matmul_dyn_vv_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{5, 8, {7}};
    auto l0  = mm->add_parameter("1", migraphx::shape{migraphx::shape::float_type, {dd}});
    auto l1  = mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {dd}});
    auto sl0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l0);
    auto sl1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l1);
    auto res =
        migraphx::add_apply_alpha_beta(*mm, {sl0, sl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto sr0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), res);
    auto ret = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sr0);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = dd;
    auto prog                     = read_onnx("matmul_dyn_vv_test.onnx", options);

    EXPECT(p == prog);
}
