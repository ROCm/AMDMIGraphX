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
#include <migraphx/kernels/gather_view.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/nontemporal.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// The rows of a {4, 8} table selected by two indices, read through a
// gathered view: the shape is the gathered shape with the strides of the
// data and the elements come from the selected rows
TEST_CASE(gather_view_rows)
{
    migraphx::half data[32];
    for(migraphx::index_int i = 0; i < 32; i++)
        data[i] = migraphx::half(i);
    migraphx::int32_t indices[2] = {3, 1};
    auto d   = migraphx::make_tensor_view(data, migraphx::make_shape(migraphx::index_ints<4, 8>{}));
    auto idx = migraphx::make_tensor_view(indices, migraphx::make_shape(migraphx::index_ints<2>{}));
    auto g   = migraphx::make_gather_view<0>(d, idx);
    static_assert(decltype(g.get_shape()){}.lens[0] == 2, "The axis counts the indices");
    static_assert(decltype(g.get_shape()){}.lens[1] == 8, "The other axes are kept");
    static_assert(decltype(g.get_shape()){}.strides[0] == 8, "The data strides are kept");
    EXPECT(g.size() == 16);
    bool matches = true;
    for(migraphx::index_int i = 0; i < 2; i++)
    {
        for(migraphx::index_int j = 0; j < 8; j++)
        {
            auto expected = migraphx::half(indices[i] * 8 + j);
            if(not migraphx::float_equal(g[i * 8 + j], expected))
                matches = false;
            if(not migraphx::float_equal(g[migraphx::array<migraphx::index_int, 2>{i, j}],
                                         expected))
                matches = false;
        }
    }
    EXPECT(matches);
}

// A negative index counts from the end and an index past the end is clamped
TEST_CASE(gather_view_index_range)
{
    migraphx::half data[32];
    for(migraphx::index_int i = 0; i < 32; i++)
        data[i] = migraphx::half(i);
    migraphx::int32_t indices[2] = {-1, 9};
    auto d   = migraphx::make_tensor_view(data, migraphx::make_shape(migraphx::index_ints<4, 8>{}));
    auto idx = migraphx::make_tensor_view(indices, migraphx::make_shape(migraphx::index_ints<2>{}));
    auto g   = migraphx::make_gather_view<0>(d, idx);
    EXPECT(migraphx::float_equal(g[0], migraphx::half(24)));
    EXPECT(migraphx::float_equal(g[8], migraphx::half(24)));
}

// Vectorizing the view keeps the gather: the base tensor keeps the data
// length along the gather axis while the strides count vectors, and slicing
// the view at a gathered row gives a plain view into that row
TEST_CASE(gather_view_vectorize)
{
    migraphx::half data[32];
    for(migraphx::index_int i = 0; i < 32; i++)
        data[i] = migraphx::half(i);
    migraphx::int32_t indices[2] = {2, 0};
    auto d   = migraphx::make_tensor_view(data, migraphx::make_shape(migraphx::index_ints<4, 8>{}));
    auto idx = migraphx::make_tensor_view(indices, migraphx::make_shape(migraphx::index_ints<2>{}));
    auto g   = migraphx::make_gather_view<0>(migraphx::as_const(d), idx);
    auto v   = migraphx::vectorize_tensor<4, 1, true>(g);
    static_assert(decltype(v.get_shape()){}.lens[0] == 2, "Gathered shape");
    static_assert(decltype(v.get_shape()){}.lens[1] == 2, "Two vectors per row");
    static_assert(decltype(v.get_shape()){}.strides[0] == 2, "Row stride counts vectors");
    static_assert(decltype(v.base.get_shape()){}.lens[0] == 4, "The base keeps the data rows");
    auto row = migraphx::make_tensor_view(&v[migraphx::array<migraphx::index_int, 2>{0, 0}],
                                          migraphx::make_shape(migraphx::index_ints<2>{}));
    auto x   = migraphx::load_element(row, 1);
    EXPECT(migraphx::float_equal(x[0], migraphx::half(20)));
    EXPECT(migraphx::float_equal(x[3], migraphx::half(23)));
}
