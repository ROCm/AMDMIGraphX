/*
* The MIT License (MIT)
*
* Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(where_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto lc  = mm->add_parameter("c", migraphx::shape{migraphx::shape::bool_type, {2}});
    auto lx  = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2, 2}});
    auto ly  = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {2, 1, 2, 2}});

    auto lccm =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), lc);
    auto lxm =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), lx);
    auto lym =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 2, 2}}}), ly);

    auto r = mm->add_instruction(migraphx::make_op("where"), lccm, lxm, lym);
    mm->add_return({r});

    auto prog = migraphx::parse_onnx("where_test.onnx");

    EXPECT(p == prog);
}
