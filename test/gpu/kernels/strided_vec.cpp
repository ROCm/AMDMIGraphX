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
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/nontemporal.hpp>
#include <migraphx/kernels/test.hpp>
#include <migraphx/kernels/float_equal.hpp>

// A buffer of S interleaved sequences read as a strided view per phase: the
// loaded vectors hold every S-th element starting at the view pointer, for
// every phase of the pointer within the aligned block
template <migraphx::index_int N, migraphx::index_int S, migraphx::index_int L>
__device__ bool strided_load_matches()
{
    constexpr migraphx::index_int size = L * S;
    alignas(16) migraphx::array<migraphx::half, size> buffer;
    for(migraphx::index_int i = 0; i < size; i++)
        buffer[i] = migraphx::half(i);
    bool matches = true;
    for(migraphx::index_int phase = 0; phase < S; phase++)
    {
        auto view = migraphx::make_tensor_view(
            &buffer[phase],
            migraphx::make_shape(migraphx::index_ints<L>{}, migraphx::index_ints<S>{}));
        auto v = migraphx::vectorize_tensor<N, 0, true>(migraphx::as_const(view));
        static_assert(decltype(v.get_shape()){}.lens[0] == L / N, "Axis counts vectors");
        static_assert(decltype(v.get_shape()){}.strides[0] == 1, "Vectors are adjacent");
        for(migraphx::index_int k = 0; k < L / N; k++)
        {
            auto x = migraphx::load_element(v, k);
            static_assert(migraphx::vec_size<decltype(x)>() == N, "Loaded as the lanes");
            for(migraphx::index_int i = 0; i < N; i++)
            {
                if(not migraphx::float_equal(x[i], migraphx::half(phase + (k * N + i) * S)))
                    matches = false;
            }
        }
    }
    return matches;
}

TEST_CASE(strided_vec_load_stride2)
{
    EXPECT(strided_load_matches<8, 2, 32>());
    EXPECT(strided_load_matches<32, 2, 64>());
}

TEST_CASE(strided_vec_load_stride4)
{
    EXPECT(strided_load_matches<4, 4, 16>());
    EXPECT(strided_load_matches<8, 4, 16>());
}

// A strided view along the inner axis counts the other strides in vectors
TEST_CASE(strided_vec_shape_step)
{
    constexpr auto s =
        migraphx::make_shape(migraphx::index_ints<4, 64>{}, migraphx::index_ints<128, 2>{});
    constexpr auto v = migraphx::shape_strided_step<8, 2>(s, migraphx::_c<1>);
    static_assert(v.lens[0] == 4 and v.lens[1] == 8, "The axis counts vectors");
    static_assert(v.strides[0] == 8 and v.strides[1] == 1, "The outer stride counts vectors");
    EXPECT(v.elements() == 32);
}

// A strided vector is detected, sized by its lanes and loaded as a plain
// vector, while spanning the whole block in memory
TEST_CASE(strided_vec_type_traits)

{
    using sv = migraphx::strided_vec<migraphx::half, 8, 2>;
    static_assert(migraphx::is_strided_vec<sv>{}, "strided_vec is detected");
    static_assert(not migraphx::is_strided_vec<migraphx::vec<migraphx::half, 8>>{},
                  "vec is not strided");
    static_assert(migraphx::vec_size<sv>() == 8, "The lanes are the vector size");
    static_assert(
        migraphx::is_same<migraphx::load_type<const sv>, migraphx::vec<migraphx::half, 8>>{},
        "Loaded as a plain vector");
    static_assert(sizeof(sv) == 16 * sizeof(migraphx::half), "Spans the block");
    EXPECT(true);
}
