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

// Slice on the innermost axis followed by squeeze, similar to the
// coordinate extraction pattern in gridsample.
struct test_reorder_reshape_slice_squeeze : verify_program<test_reorder_reshape_slice_squeeze>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {1, 2, 4, 2}};
        auto input = mm->add_parameter("input", s);
        auto slc0  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto slc1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto sq0 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), slc0);
        auto sq1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), slc1);

        auto sum = mm->add_instruction(migraphx::make_op("add"), sq0, sq1);
        mm->add_instruction(migraphx::make_op("mul"), sum, sq0);
        return p;
    }
};
