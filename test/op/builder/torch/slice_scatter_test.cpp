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

#include <cstdint>
#include <vector>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

// tm::slice_scatter scatters src into the [start:end:step] slice along `dim`; the
// scatter indices carry the resolved position of each src element along that dim.
TEST_CASE(torch_kit_slice_scatter_op_builder_test)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module mm;
    auto input = mm.add_parameter("input", {f, {4, 3}});
    auto src   = mm.add_parameter("src", {f, {2, 3}});

    std::vector<int64_t> idx_data = {0, 0, 0, 1, 1, 1};
    auto indices                  = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2, 3}}, idx_data});
    auto std_input = mm.add_instruction(migraphx::make_op("contiguous"), input);
    auto std_src   = mm.add_instruction(migraphx::make_op("contiguous"), src);
    mm.add_instruction(
        migraphx::make_op("scatter_none", {{"axis", 0}}), std_input, indices, std_src);

    migraphx::value options{{"dim", 0}, {"start", 0}, {"end", 2}, {"step", 1}};
    EXPECT(mm == make_op_module("tm::slice_scatter", options, mm.get_parameters()));
}
