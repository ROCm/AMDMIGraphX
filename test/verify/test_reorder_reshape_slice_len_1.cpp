/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

#include <numeric>

struct test_reorder_reshape_slice_len_1 : verify_program<test_reorder_reshape_slice_len_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 128, 3}};
        std::vector<float> data(s.elements());
        std::iota(data.begin(), data.end(), 0);
        auto lit   = mm->add_literal(migraphx::literal{s, data});
        auto input = mm->add_parameter("input", s);
        auto x     = mm->add_instruction(migraphx::make_op("add"), input, lit);
        auto slc0  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), x);
        auto slc1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), x);
        auto slc2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), x);

        std::vector<int64_t> lens = {1, 128};
        auto r0 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), slc0);
        auto r1 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), slc1);
        auto r2 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", lens}}), slc2);

        auto sum = mm->add_instruction(migraphx::make_op("add"), r0, r1);
        auto ret = mm->add_instruction(migraphx::make_op("mul"), sum, r2);
        mm->add_return({ret});
        return p;
    }
};
