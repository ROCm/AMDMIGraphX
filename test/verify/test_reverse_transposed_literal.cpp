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
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <numeric>

// reverse over a transposed constant, which constant-folds into a non-standard
// literal that reverse must index by its logical coordinates
struct test_reverse_transposed_literal : verify_program<test_reverse_transposed_literal>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
        std::vector<float> data(s.elements());
        std::iota(data.begin(), data.end(), 1.0f);
        auto lit = mm->add_literal(migraphx::literal{s, data});

        std::vector<int64_t> perm = {0, 2, 1};
        auto tr = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", perm}}), lit);

        std::vector<int64_t> axes = {1};
        auto rev = mm->add_instruction(migraphx::make_op("reverse", {{"axes", axes}}), tr);

        auto x = mm->add_parameter("x",
                                   migraphx::shape{migraphx::shape::float_type, {2, 4, 3}});
        mm->add_instruction(migraphx::make_op("add"), rev, x);
        return p;
    }
};
