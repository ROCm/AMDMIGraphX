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
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

// These tests verify the deref operation by using addressof to generate
// valid pointers at runtime, then dereferencing them back to the original values.
// This pattern works across all targets (ref, CPU, GPU) because the pointers
// are generated on each target's memory space.

template <migraphx::shape::type_t DType>
struct test_deref : verify_program<test_deref<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{DType, {2, 4}};
        auto param = mm->add_parameter("x", s);
        auto addrs = mm->add_instruction(migraphx::make_op("addressof"), param);
        mm->add_instruction(
            migraphx::make_op("deref", {{"target_type", migraphx::to_value(DType)}}), addrs);
        return p;
    }
};

template struct test_deref<migraphx::shape::float_type>;
template struct test_deref<migraphx::shape::half_type>;
template struct test_deref<migraphx::shape::double_type>;

struct test_deref_int32 : verify_program<test_deref_int32>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::int32_type, {8}};
        auto param = mm->add_parameter("x", s);
        auto addrs = mm->add_instruction(migraphx::make_op("addressof"), param);
        mm->add_instruction(
            migraphx::make_op("deref", {{"target_type", migraphx::shape::int32_type}}), addrs);
        return p;
    }
};

struct test_deref_3d : verify_program<test_deref_3d>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
        auto param = mm->add_parameter("x", s);
        auto addrs = mm->add_instruction(migraphx::make_op("addressof"), param);
        mm->add_instruction(
            migraphx::make_op("deref", {{"target_type", migraphx::shape::float_type}}), addrs);
        return p;
    }
};

// Note: Standalone addressof verify tests are not included because they compare
// absolute memory addresses between CPU (ref) and GPU targets, which are in
// different address spaces and will never match. The deref roundtrip tests above
// verify that addressof works correctly by testing deref(addressof(x)) == x.
