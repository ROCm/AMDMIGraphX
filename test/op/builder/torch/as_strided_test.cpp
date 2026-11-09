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

// tm::as_strided materializes a strided view by flattening the input and gathering
// the element at storage_offset + strided.index(i) for every output coordinate.
TEST_CASE(torch_kit_as_strided_op_builder_test)
{
    const auto f = migraphx::shape::float_type;

    migraphx::module mm;
    auto x = mm.add_parameter("x", {f, {4}});

    std::vector<int64_t> idx_data = {0, 1, 2, 3};
    auto indices                  = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {4}}, idx_data});
    auto flat_inp = mm.add_instruction(migraphx::make_op("contiguous"), x);
    flat_inp      = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {-1}}}), flat_inp);
    auto gathered =
        mm.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), flat_inp, indices);
    mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2}}}), gathered);

    migraphx::value options{{"size", {2, 2}}, {"stride", {2, 1}}, {"storage_offset", 0}};
    EXPECT(mm == make_op_module("tm::as_strided", options, mm.get_parameters()));
}

// size and stride must have matching lengths.
TEST_CASE(torch_kit_as_strided_size_stride_mismatch)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    mm.add_parameter("x", {f, {4}});
    EXPECT(test::throws<migraphx::exception>([&] {
        make_op_module("tm::as_strided",
                       {{"size", {2, 2}}, {"stride", {2}}, {"storage_offset", 0}},
                       mm.get_parameters());
    }));
}
