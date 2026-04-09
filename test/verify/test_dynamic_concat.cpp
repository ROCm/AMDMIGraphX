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
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/sym.hpp>

struct test_dynamic_concat_axis0 : verify_program<test_dynamic_concat_axis0>
{
    migraphx::program create_program() const
    {
        migraphx::shape s0{migraphx::shape::float_type, {{2, 4, {2}}, {2, 3, {2}}}};
        migraphx::shape s1{migraphx::shape::float_type, {{3, 4, {4}}, {2, 3, {2}}}};
        migraphx::shape s2{migraphx::shape::float_type, {{1, 5, {3}}, {2, 3, {2}}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("X", s0);
        auto y   = mm->add_parameter("Y", s1);
        auto z   = mm->add_parameter("Z", s2);
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y, z);
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"X", migraphx::shape{migraphx::shape::float_type, {2, 2}}},
                {"Y", migraphx::shape{migraphx::shape::float_type, {3, 2}}},
                {"Z", migraphx::shape{migraphx::shape::float_type, {1, 2}}}};
    }
};

struct test_symbolic_concat_axis0_gpu : verify_program<test_symbolic_concat_axis0_gpu>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto n  = var("n");
        auto d0 = var("d0");
        auto d1 = var("d1");
        auto d2 = var("d2");
        using dd = migraphx::shape::dynamic_dimension;

        migraphx::shape s0{migraphx::shape::float_type, {dd{2, 4, {}, d0}, dd{2, 3, {}, n}}};
        migraphx::shape s1{migraphx::shape::float_type, {dd{3, 4, {}, d1}, dd{2, 3, {}, n}}};
        migraphx::shape s2{migraphx::shape::float_type, {dd{1, 5, {}, d2}, dd{2, 3, {}, n}}};
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("X", s0);
        auto y   = mm->add_parameter("Y", s1);
        auto z   = mm->add_parameter("Z", s2);
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), x, y, z);
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"X", migraphx::shape{migraphx::shape::float_type, {2, 2}}},
                {"Y", migraphx::shape{migraphx::shape::float_type, {3, 2}}},
                {"Z", migraphx::shape{migraphx::shape::float_type, {1, 2}}}};
    }
};
