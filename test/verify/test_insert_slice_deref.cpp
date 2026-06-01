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

// insert_slice(source, dest, deref_dest=true) writes source through the pointers in dest.
// Use a single parameter x: dest = addressof(x), source = x, then insert_slice(x, dest) and
// deref(insert_slice output). Pointers are generated on each target (addressof(x)), so ref gets
// host pointers and gpu gets device pointers — same pattern as test_deref.cpp.
template <migraphx::shape::type_t DType>
struct test_insert_slice_deref : verify_program<test_insert_slice_deref<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 4}};
        migraphx::shape s2{DType, {4, 4}};
        auto x   = mm->add_parameter("x", s);
        x = mm->add_instruction(migraphx::make_op("add"), x, x);
        auto y = mm->add_parameter("y", s2);
        auto dest = mm->add_instruction(migraphx::make_op("addressof"), y);
        auto out = mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_offsets", {1, 0}},
                              {"static_strides", {1, 1}},
                              {"deref_dest", true}}),
            x,
            dest);
        auto deref = mm->add_instruction(
            migraphx::make_op("deref", {{"target_type", migraphx::to_value(DType)}}), out);
        return p;
    }
};

template struct test_insert_slice_deref<migraphx::shape::float_type>;
template struct test_insert_slice_deref<migraphx::shape::half_type>;


template <migraphx::shape::type_t DType>
struct test_insert_slice_deref_offset : verify_program<test_insert_slice_deref_offset<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 4}};
        migraphx::shape s2{DType, {4, 4}};
        migraphx::shape offsets_shape{migraphx::shape::int64_type, {2}};
        std::vector<int64_t> offsets_data = {0, 1};
        auto offsets_lit =
            mm->add_literal(migraphx::literal{offsets_shape, offsets_data});
        auto x   = mm->add_parameter("x", s);
        x = mm->add_instruction(migraphx::make_op("add"), x, x);
        auto y = mm->add_parameter("y", s2);
        auto dest = mm->add_instruction(migraphx::make_op("addressof"), y);
        auto out = mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {
                              {"static_strides", {1, 1}},
                              {"deref_dest", true}}),
            x,
            dest,
            offsets_lit);
        auto deref = mm->add_instruction(
            migraphx::make_op("deref", {{"target_type", migraphx::to_value(DType)}}), out);
        return p;
    }
};

template struct test_insert_slice_deref_offset<migraphx::shape::float_type>;
template struct test_insert_slice_deref_offset<migraphx::shape::half_type>;
