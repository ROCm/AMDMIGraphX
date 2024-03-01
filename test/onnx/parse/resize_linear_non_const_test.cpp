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

< < < < < < < < HEAD : test / onnx / parse / expand_dyn_test.cpp TEST_CASE(expand_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape x_shape(migraphx::shape::float_type, {3, 1, 1});
    migraphx::shape dims_shape(migraphx::shape::int64_type, {4});
    auto x_param    = mm->add_parameter("x", x_shape);
    auto dims_param = mm->add_parameter("dims", dims_shape);
    mm->add_instruction(migraphx::make_op("broadcast_with_dims"), x_param, dims_param);
    auto prog = optimize_onnx("expand_dyn_test.onnx");
    EXPECT(p == prog);
    == == == == TEST_CASE(resize_linear_non_const_test)
    {
        // runtime (non-constant) input is only supported in "nearest" mode
        migraphx::onnx_options options;
        EXPECT(test::throws([&] { parse_onnx("resize_linear_non_const_test.onnx", options); }));
        >>>>>>>> c14e3c8ff744369c84d603662fd36c1ab352228a : test / onnx / parse /
                                                            resize_linear_non_const_test.cpp
    }
