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

#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>

// tm::index_copy reshapes the 1-D index to the src rank, broadcasts it to the src
// shape, and scatters src into the rows of `dim` it selects.
TEST_CASE(torch_kit_index_copy_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;

    migraphx::module mm;
    auto inp = mm.add_parameter("inp", {f, {5, 4}});
    auto idx = mm.add_parameter("idx", {i, {2}});
    auto src = mm.add_parameter("src", {f, {2, 4}});

    auto scatter_idx = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 1}}}), idx);
    scatter_idx = mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4}}}),
                                     scatter_idx);
    mm.add_instruction(migraphx::make_op("scatter_none", {{"axis", 0}}), inp, scatter_idx, src);

    EXPECT(mm == make_op_module("tm::index_copy", {{"dim", 0}}, mm.get_parameters()));
}
