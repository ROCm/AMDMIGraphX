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
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

// insert_slice(source, dest): copies dest to output then scatters source at
// dest_idx = src_idx * strides + offsets. With offsets={0,0}, strides={1,1},
// output becomes source. Verifies ref and gpu match.
template <migraphx::shape::type_t DType>
struct test_insert_slice : verify_program<test_insert_slice<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 4}};
        auto source = mm->add_parameter("source", s);
        auto dest   = mm->add_parameter("dest", s);
        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_offsets", {0, 0}},
                              {"static_strides", {1, 1}},
                              {"deref_dest", false}}),
            source,
            dest);
        return p;
    }
};

template struct test_insert_slice<migraphx::shape::float_type>;
template struct test_insert_slice<migraphx::shape::half_type>;

// Non-zero offset on axis 0: dest (3,4), source (2,4), offsets {1,0}. Source fills output[1:3,:].
template <migraphx::shape::type_t DType>
struct test_insert_slice_offset_axis0 : verify_program<test_insert_slice_offset_axis0<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape src_shape{DType, {2, 4}};
        migraphx::shape dest_shape{DType, {3, 4}};
        auto source = mm->add_parameter("source", src_shape);
        auto dest   = mm->add_parameter("dest", dest_shape);
        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_offsets", {1, 0}},
                              {"static_strides", {1, 1}},
                              {"deref_dest", false}}),
            source,
            dest);
        return p;
    }
};

template struct test_insert_slice_offset_axis0<migraphx::shape::float_type>;
template struct test_insert_slice_offset_axis0<migraphx::shape::half_type>;

// Non-zero offset on axis 1: dest (2,5), source (2,4), offsets {0,1}. Source fills output[:,1:5].
template <migraphx::shape::type_t DType>
struct test_insert_slice_offset_axis1 : verify_program<test_insert_slice_offset_axis1<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape src_shape{DType, {2, 4}};
        migraphx::shape dest_shape{DType, {2, 5}};
        auto source = mm->add_parameter("source", src_shape);
        auto dest   = mm->add_parameter("dest", dest_shape);
        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_offsets", {0, 1}},
                              {"static_strides", {1, 1}},
                              {"deref_dest", false}}),
            source,
            dest);
        return p;
    }
};

template struct test_insert_slice_offset_axis1<migraphx::shape::float_type>;
template struct test_insert_slice_offset_axis1<migraphx::shape::half_type>;

// Non-zero offsets on both axes: dest (4, 5), source (2, 3), offsets {1, 2}. Source fills output[1:3, 2:5].
template <migraphx::shape::type_t DType>
struct test_insert_slice_offset_both_axes
    : verify_program<test_insert_slice_offset_both_axes<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape src_shape{DType, {2, 3}};
        migraphx::shape dest_shape{DType, {4, 5}};
        auto source = mm->add_parameter("source", src_shape);
        auto dest   = mm->add_parameter("dest", dest_shape);
        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_offsets", {1, 2}},
                              {"static_strides", {1, 1}},
                              {"deref_dest", false}}),
            source,
            dest);
        return p;
    }
};

template struct test_insert_slice_offset_both_axes<migraphx::shape::float_type>;
template struct test_insert_slice_offset_both_axes<migraphx::shape::half_type>;

// Offsets provided as input (literal) instead of static_offsets: dest (2,5), source (2,4),
// offsets literal [0, 1]. Source fills output[:,1:5]. Uses static_strides only.
template <migraphx::shape::type_t DType>
struct test_insert_slice_offsets_input : verify_program<test_insert_slice_offsets_input<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape src_shape{DType, {2, 4}};
        migraphx::shape dest_shape{DType, {2, 5}};
        migraphx::shape offsets_shape{migraphx::shape::int64_type, {2}};
        std::vector<int64_t> offsets_data = {0, 1};

        auto source = mm->add_parameter("source", src_shape);
        auto dest   = mm->add_parameter("dest", dest_shape);
        auto offsets_lit =
            mm->add_literal(migraphx::literal{offsets_shape, offsets_data});

        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_strides", {1, 1}}, {"deref_dest", false}}),
            source,
            dest,
            offsets_lit);
        return p;
    }
};

template struct test_insert_slice_offsets_input<migraphx::shape::float_type>;
template struct test_insert_slice_offsets_input<migraphx::shape::half_type>;

// Batched offsets [batch, rank]: row b gives per-axis offsets for batch index b (source axis 0).
template <migraphx::shape::type_t DType>
struct test_insert_slice_batched_offsets : verify_program<test_insert_slice_batched_offsets<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape src_shape{DType, {2, 2}};
        migraphx::shape dest_shape{DType, {2, 3}};
        migraphx::shape offsets_shape{migraphx::shape::int64_type, {2, 2}};
        // batch 0 -> {0,0}, batch 1 -> {0,1}; places 2x2 src blocks at (0,0) and (1,1) corners.
        std::vector<int64_t> offsets_data = {0, 0, 0, 1};

        auto source = mm->add_parameter("source", src_shape);
        auto dest   = mm->add_parameter("dest", dest_shape);
        auto offsets_lit =
            mm->add_literal(migraphx::literal{offsets_shape, offsets_data});

        mm->add_instruction(
            migraphx::make_op("insert_slice",
                             {{"static_strides", {1, 1}}, {"deref_dest", false}}),
            source,
            dest,
            offsets_lit);
        return p;
    }
};

template struct test_insert_slice_batched_offsets<migraphx::shape::float_type>;
template struct test_insert_slice_batched_offsets<migraphx::shape::half_type>;
