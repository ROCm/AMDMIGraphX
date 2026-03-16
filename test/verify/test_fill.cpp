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

template <migraphx::shape::type_t DType>
struct test_fill : verify_program<test_fill<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{DType, {1}, {0}};
        migraphx::shape data_shape{DType, {3, 4, 4}};
        auto value = mm->add_parameter("value", scalar_shape);
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};

template struct test_fill<migraphx::shape::float_type>;
template struct test_fill<migraphx::shape::half_type>;
template struct test_fill<migraphx::shape::bf16_type>;
template struct test_fill<migraphx::shape::int32_type>;
template struct test_fill<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_fill<migraphx::shape::fp8e4m3fn_type>;

template <migraphx::shape::type_t DType>
struct test_fill_large : verify_program<test_fill_large<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{DType, {1}, {0}};
        migraphx::shape data_shape{DType, {64, 128, 128}};
        auto value = mm->add_parameter("value", scalar_shape);
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};

template struct test_fill_large<migraphx::shape::float_type>;
template struct test_fill_large<migraphx::shape::half_type>;
template struct test_fill_large<migraphx::shape::bf16_type>;

template <migraphx::shape::type_t DType>
struct test_fill_literal : verify_program<test_fill_literal<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{DType, {1}, {0}};
        migraphx::shape data_shape{DType, {2, 3, 4}};
        auto value = mm->add_literal(migraphx::literal{scalar_shape, {7}});
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};

template struct test_fill_literal<migraphx::shape::float_type>;
template struct test_fill_literal<migraphx::shape::int32_type>;

struct test_fill_single_element : verify_program<test_fill_single_element>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{migraphx::shape::float_type, {1}, {0}};
        migraphx::shape data_shape{migraphx::shape::float_type, {1}};
        auto value = mm->add_parameter("value", scalar_shape);
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};

struct test_fill_1d : verify_program<test_fill_1d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{migraphx::shape::float_type, {1}, {0}};
        migraphx::shape data_shape{migraphx::shape::float_type, {256}};
        auto value = mm->add_parameter("value", scalar_shape);
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};

struct test_fill_2d : verify_program<test_fill_2d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape scalar_shape{migraphx::shape::float_type, {1}, {0}};
        migraphx::shape data_shape{migraphx::shape::float_type, {16, 32}};
        auto value = mm->add_parameter("value", scalar_shape);
        auto data  = mm->add_parameter("data", data_shape);
        mm->add_instruction(migraphx::make_op("fill"), value, data);
        return p;
    }
};
