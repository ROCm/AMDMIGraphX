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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

// Concat of a producer seen through a reshape whose inverse cannot alias the
// concat buffer (the slice is strided across the merged dims), so
// eliminate_concat must fall back to copying instead of eliding
template <migraphx::shape::type_t DType>
struct test_concat_view_chain_copy : verify_program<test_concat_view_chain_copy<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s1{DType, {4}};
        migraphx::shape s2{DType, {2, 3}};
        auto x  = mm->add_parameter("x", s1);
        auto xs = mm->add_parameter("xs", s1);
        auto y  = mm->add_parameter("y", s2);
        auto ys = mm->add_parameter("ys", s2);
        auto x2 = mm->add_instruction(migraphx::make_op("mul"), x, xs);
        auto y2 = mm->add_instruction(migraphx::make_op("mul"), y, ys);
        auto r  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), x2);
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), r, y2);
        return p;
    }
};

template struct test_concat_view_chain_copy<migraphx::shape::float_type>;
template struct test_concat_view_chain_copy<migraphx::shape::half_type>;
