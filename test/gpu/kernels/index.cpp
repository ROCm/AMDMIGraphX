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
 *
 */
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/type_traits.hpp>
#include <migraphx/kernels/test.hpp>

// The test harness runs a single thread, so emulate a whole group by running
// its strided loop serially: the union of every thread's share is [0, n).
struct serial_group
{
    migraphx::index idx;

    template <class N, class F>
    constexpr void local_stride(N n, F f) const
    {
        for(migraphx::index_int i = 0; i < n; i++)
            f(i);
    }
};

// Count how many times block_stride visits each element and expect exactly one
// visit per element. Regression for a tail loop that iterated [m, m + Block)
// instead of [m * Block, n), revisiting earlier elements and skipping the ones
// past the last full block.
template <migraphx::index_int Block, migraphx::index_int N, bool RuntimeSize = false>
TEST_CASE_TEMPLATE(check_block_stride)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    migraphx::conditional_t<RuntimeSize, migraphx::index_int, migraphx::index_constant<N>> n =
        migraphx::_c<N>;
    migraphx::array<int, N> counts{};
    migraphx::block_stride<serial_group, Block>(idx, n)([&](auto i) {
        EXPECT(i < n);
        counts[i]++;
    });
    EXPECT(migraphx::all_of(counts.begin(), counts.end(), [](int c) { return c == 1; }));
}

TEST_CASE_REGISTER(check_block_stride<8, 64>);
TEST_CASE_REGISTER(check_block_stride<8, 60>);
TEST_CASE_REGISTER(check_block_stride<8, 60, true>);
TEST_CASE_REGISTER(check_block_stride<8, 5>);
TEST_CASE_REGISTER(check_block_stride<8, 1>);
TEST_CASE_REGISTER(check_block_stride<1, 7>);
TEST_CASE_REGISTER(check_block_stride<4, 33>);
