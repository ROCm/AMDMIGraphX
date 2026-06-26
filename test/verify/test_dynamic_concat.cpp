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
#include <migraphx/shape.hpp>
#include <migraphx/generate.hpp>

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

// KV-cache style: dynamic past + static present concatenated on axis 2 (half_type).
struct test_dynamic_concat_kv_cache_axis2 : verify_program<test_dynamic_concat_kv_cache_axis2>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto psl = var("psl", {1, 64});
        using dd = migraphx::shape::dynamic_dimension;

        migraphx::shape past_shape{migraphx::shape::half_type, {dd{1, 1}, dd{5, 5}, dd{psl}, dd{64, 64}}};
        migraphx::shape current_shape{migraphx::shape::half_type, {1, 5, 1, 64}};

        migraphx::program p;
        auto* mm = p.get_main_module();
        auto past_key   = mm->add_parameter("past_key_values.0.key", past_shape);
        auto current_key = mm->add_literal(migraphx::generate_literal(current_shape));
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), past_key, current_key);
        return p;
    }

    std::unordered_map<std::string, migraphx::shape> get_test_dims() const
    {
        return {{"past_key_values.0.key",
                 migraphx::shape{migraphx::shape::half_type, {1, 5, 1, 64}}}};
    }
};

struct test_symbolic_concat_axis0_gpu : verify_program<test_symbolic_concat_axis0_gpu>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;
        auto n  = var("n", {2, 3});
        auto d0 = var("d0", {2, 4});
        auto d1 = var("d1", {3, 4});
        auto d2 = var("d2", {1, 5});
        using dd = migraphx::shape::dynamic_dimension;

        migraphx::shape s0{migraphx::shape::float_type, {dd{d0}, dd{n}}};
        migraphx::shape s1{migraphx::shape::float_type, {dd{d1}, dd{n}}};
        migraphx::shape s2{migraphx::shape::float_type, {dd{d2}, dd{n}}};
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
