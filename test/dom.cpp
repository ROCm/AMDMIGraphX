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
#include <migraphx/dom_info.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <test.hpp>

TEST_CASE(dom1)
{
    migraphx::module mm;
    auto ins1 = mm.add_parameter("entry", {migraphx::shape::float_type});
    auto ins2 = mm.add_instruction(pass_op{}, ins1);
    auto ins3 = mm.add_instruction(pass_op{}, ins2);
    auto ins4 = mm.add_instruction(pass_op{}, ins2);
    auto ins5 = mm.add_instruction(pass_op{}, ins3, ins4);
    auto ins6 = mm.add_instruction(pass_op{}, ins2);

    auto dom = migraphx::compute_dominator(mm);
    EXPECT(dom.strictly_dominate(ins1, ins2));
    EXPECT(dom.strictly_dominate(ins2, ins3));
    EXPECT(dom.strictly_dominate(ins2, ins4));
    EXPECT(dom.strictly_dominate(ins2, ins5));
    EXPECT(dom.strictly_dominate(ins2, ins6));

    EXPECT(not dom.strictly_dominate(ins3, ins6));
    EXPECT(not dom.strictly_dominate(ins4, ins6));
    EXPECT(not dom.strictly_dominate(ins3, ins5));
    EXPECT(not dom.strictly_dominate(ins4, ins5));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
