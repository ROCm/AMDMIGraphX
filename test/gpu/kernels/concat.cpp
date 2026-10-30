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
#include <migraphx/kernels/concat.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/float_equal.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/test.hpp>

using migraphx::index_int;
using migraphx::index_ints;
using migraphx::make_shape;
using migraphx::make_tensor_view;
using migraphx::concat::layout;
using migraphx::op::id;

// A packed view over a local buffer with the given lens
template <index_int... Lens, class T, index_int N>
__device__ auto view_of(migraphx::array<T, N>& buffer)
{
    static_assert(N == (Lens * ...), "Buffer holds the view");
    return make_tensor_view(buffer.data(), make_shape(index_ints<Lens...>{}));
}

template <class T, index_int N>
__device__ void fill_iota(migraphx::array<T, N>& buffer, T start)
{
    migraphx::iota(buffer.begin(), buffer.end(), start);
}

struct add2
{
    template <class A, class B>
    __device__ auto operator()(A a, B b) const
    {
        return a + b;
    }
};

struct twice
{
    template <class A>
    __device__ auto operator()(A a) const
    {
        return a * 2;
    }
};

TEST_CASE(pack_end_each_axis)
{
    migraphx::array<float, 48> buffer{};
    auto x     = view_of<2, 3, 8>(buffer);
    using pack = decltype(migraphx::pack(id{}, x));
    static_assert(migraphx::concat::pack_end<0, pack>() == 2, "Extent along axis 0");
    static_assert(migraphx::concat::pack_end<1, pack>() == 3, "Extent along axis 1");
    static_assert(migraphx::concat::pack_end<2, pack>() == 8, "Extent along axis 2");
    EXPECT(migraphx::concat::pack_end<1, pack>() == 3);
}

// The extent comes from the first input of a fused pack, not the extra inputs
TEST_CASE(pack_end_fused_pack)
{
    migraphx::array<float, 48> buffer{};
    migraphx::array<float, 16> extra{};
    auto x     = view_of<2, 3, 8>(buffer);
    auto w     = view_of<1, 2, 8>(extra);
    using pack = decltype(migraphx::pack(add2{}, x, w));
    static_assert(migraphx::concat::pack_end<1, pack>() == 3, "Extent of the first input");
    EXPECT(migraphx::concat::pack_end<1, pack>() == 3);
}

TEST_CASE(concat_offsets_exclusive_scan)
{
    constexpr migraphx::array<index_int, 4> ends = {3, 5, 2, 4};
    constexpr auto offsets                       = migraphx::concat::concat_offsets(ends);
    static_assert(offsets == migraphx::array<index_int, 4>{0, 3, 8, 10}, "Starts of each input");
    EXPECT(offsets == migraphx::array<index_int, 4>{0, 3, 8, 10});
}

TEST_CASE(concat_offsets_single_input)
{
    constexpr migraphx::array<index_int, 1> ends = {7};
    constexpr auto offsets                       = migraphx::concat::concat_offsets(ends);
    static_assert(offsets == migraphx::array<index_int, 1>{0}, "Single input starts at zero");
    EXPECT(offsets == migraphx::array<index_int, 1>{0});
}

TEST_CASE(layout_mixed_extents)
{
    migraphx::array<float, 48> b0{};
    migraphx::array<float, 80> b1{};
    migraphx::array<float, 32> b2{};
    auto x0 = view_of<2, 3, 8>(b0);
    auto x1 = view_of<2, 5, 8>(b1);
    auto x2 = view_of<2, 2, 8>(b2);
    using l = layout<1,
                     decltype(migraphx::pack(id{}, x0)),
                     decltype(migraphx::pack(id{}, x1)),
                     decltype(migraphx::pack(id{}, x2))>;
    static_assert(l::ends == migraphx::array<index_int, 3>{3, 5, 2}, "Extents along the axis");
    static_assert(l::offsets == migraphx::array<index_int, 3>{0, 3, 8}, "Start of each input");
    static_assert(l::max_size == 5, "Largest extent");
    EXPECT(l::ends == migraphx::array<index_int, 3>{3, 5, 2});
    EXPECT(l::offsets == migraphx::array<index_int, 3>{0, 3, 8});
    EXPECT(l::max_size == 5);
}

TEST_CASE(layout_single_input)
{
    migraphx::array<float, 48> b0{};
    auto x0 = view_of<2, 3, 8>(b0);
    using l = layout<2, decltype(migraphx::pack(id{}, x0))>;
    static_assert(l::ends == migraphx::array<index_int, 1>{8}, "Extent along the last axis");
    static_assert(l::offsets == migraphx::array<index_int, 1>{0}, "Starts at zero");
    static_assert(l::max_size == 8, "Largest extent");
    EXPECT(l::max_size == 8);
}

// The slice has the input's lens, the output's strides and starts at the offset along the axis
TEST_CASE(concat_slice_view)
{
    migraphx::array<float, 128> out_buffer{};
    migraphx::array<float, 48> in_buffer{};
    auto out = view_of<2, 8, 8>(out_buffer);
    auto in  = view_of<2, 3, 8>(in_buffer);
    auto s   = migraphx::concat::concat_slice<1>(out, in, migraphx::_c<5>);
    EXPECT(s.get_shape().lens == migraphx::array<index_int, 3>{2, 3, 8});
    EXPECT(s.get_shape().strides == migraphx::array<index_int, 3>{64, 8, 1});
    EXPECT(s.data() == out.data() + 5 * 8);
}

// Concatenates two iota-filled inputs on Axis with the simple algo and checks every output
// element against the input it came from
template <index_int Axis, class Lens0, class Lens1, class LensY>
__device__ bool simple_concat_matches()
{
    constexpr auto s0 = make_shape(Lens0{});
    constexpr auto s1 = make_shape(Lens1{});
    constexpr auto sy = make_shape(LensY{});
    migraphx::array<float, s0.elements()> b0{};
    migraphx::array<float, s1.elements()> b1{};
    migraphx::array<float, sy.elements()> by{};
    fill_iota(b0, 0.0f);
    fill_iota(b1, 1000.0f);
    auto x0 = make_tensor_view(b0.data(), s0);
    auto x1 = make_tensor_view(b1.data(), s1);
    auto y  = make_tensor_view(by.data(), sy);
    migraphx::concat::run<migraphx::concat::simple, Axis>(migraphx::pack(id{}, x0),
                                                          migraphx::pack(id{}, x1))(id{}, y);
    bool matches = true;
    for(index_int i = 0; i < sy.elements(); i++)
    {
        auto m         = sy.multi(i);
        float expected = 0;
        if(m[Axis] < s0.lens[Axis])
        {
            expected = x0[m];
        }
        else
        {
            m[Axis] -= s0.lens[Axis];
            expected = x1[m];
        }
        if(not migraphx::float_equal(y[sy.multi(i)], expected))
            matches = false;
    }
    return matches;
}

TEST_CASE(run_simple_axis0)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    EXPECT(
        simple_concat_matches<0, index_ints<2, 3, 4>, index_ints<3, 3, 4>, index_ints<5, 3, 4>>());
}

TEST_CASE(run_simple_axis1)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    EXPECT(
        simple_concat_matches<1, index_ints<2, 3, 4>, index_ints<2, 5, 4>, index_ints<2, 8, 4>>());
}

TEST_CASE(run_simple_last_axis)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    EXPECT(
        simple_concat_matches<2, index_ints<2, 3, 4>, index_ints<2, 3, 1>, index_ints<2, 3, 5>>());
}

// Each pack's op combines its inputs before the slice is written, and the post op runs on the
// concatenated value
TEST_CASE(run_simple_fused_ops)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    migraphx::array<float, 12> b0{};
    migraphx::array<float, 12> w0{};
    migraphx::array<float, 8> b1{};
    migraphx::array<float, 8> w1{};
    migraphx::array<float, 20> by{};
    fill_iota(b0, 0.0f);
    fill_iota(w0, 100.0f);
    fill_iota(b1, 1000.0f);
    fill_iota(w1, 2000.0f);
    auto x0 = view_of<2, 3, 2>(b0);
    auto v0 = view_of<2, 3, 2>(w0);
    auto x1 = view_of<2, 2, 2>(b1);
    auto v1 = view_of<2, 2, 2>(w1);
    auto y  = view_of<2, 5, 2>(by);
    migraphx::concat::run<migraphx::concat::simple, 1>(migraphx::pack(add2{}, x0, v0),
                                                       migraphx::pack(add2{}, x1, v1))(twice{}, y);
    bool matches = true;
    for(index_int i = 0; i < y.get_shape().elements(); i++)
    {
        auto m         = y.get_shape().multi(i);
        float expected = m[1] < 3 ? 2 * (x0[m] + v0[m])
                                  : 2 * (x1[{m[0], m[1] - 3, m[2]}] + v1[{m[0], m[1] - 3, m[2]}]);
        if(not migraphx::float_equal(y[m], expected))
            matches = false;
    }
    EXPECT(matches);
}

// Writes each input's index over its slice of the output, so the output records which input
// run assigned to every element and the start offset it was given
template <index_int NArgs, index_int MaxSize>
struct stamp_depth
{
    struct algo
    {
        template <class Depth, class G, class... Xs>
        __device__ auto run(Depth depth, G, Xs...) const
        {
            return [=](auto z, auto, auto...) { migraphx::fill(z.begin(), z.end(), depth); };
        }

        template <class F, class... Outputs>
        __device__ void finish(F, Outputs...) const
        {
        }
    };

    template <class Info, class Output>
    static __device__ auto make(migraphx::index, Info info, Output)
    {
        static_assert(decltype(info.nargs){} == NArgs, "One pack per input");
        static_assert(decltype(info.max_size){} == MaxSize, "Largest extent along the axis");
        return algo{};
    }
};

// Inputs with extents 1, 2, 3 and 4 are visited in order with running start offsets 0, 1, 3, 6
TEST_CASE(run_visits_inputs_in_order)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    migraphx::array<float, 4> b0{};
    migraphx::array<float, 8> b1{};
    migraphx::array<float, 12> b2{};
    migraphx::array<float, 16> b3{};
    migraphx::array<float, 40> by{};
    migraphx::fill(by.begin(), by.end(), -1.0f);
    auto y = view_of<2, 10, 2>(by);
    migraphx::concat::run<stamp_depth<4, 4>, 1>(migraphx::pack(id{}, view_of<2, 1, 2>(b0)),
                                                migraphx::pack(id{}, view_of<2, 2, 2>(b1)),
                                                migraphx::pack(id{}, view_of<2, 3, 2>(b2)),
                                                migraphx::pack(id{}, view_of<2, 4, 2>(b3)))(id{},
                                                                                            y);
    constexpr migraphx::array<index_int, 4> starts = {0, 1, 3, 6};
    bool matches                                   = true;
    for(index_int i = 0; i < y.get_shape().elements(); i++)
    {
        auto m = y.get_shape().multi(i);
        // the last start not past the axis position is the input that owns it
        auto depth = migraphx::count_if(
            starts.begin(), starts.end(), [&](index_int start) { return start <= m[1]; });
        if(not migraphx::float_equal(y[m], depth - 1))
            matches = false;
    }
    EXPECT(matches);
}

// Many same-shaped inputs expand without a recursive fold over the packs
TEST_CASE(run_many_inputs)
{
    auto idx = migraphx::make_index();
    if(idx.local != 0)
        return;
    constexpr index_int n = 32;
    migraphx::array<float, 2 * n> buffer{};
    migraphx::array<float, 2 * n> by{};
    fill_iota(buffer, 0.0f);
    auto y = view_of<1, n, 2>(by);
    migraphx::sequence_c<n>([&](auto... is) {
        migraphx::concat::run<migraphx::concat::simple, 1>(migraphx::pack(
            id{}, make_tensor_view(buffer.data() + is * 2, make_shape(index_ints<1, 1, 2>{})))...)(
            id{}, y);
    });
    EXPECT(migraphx::equal(buffer.begin(), buffer.end(), by.begin(), [](float a, float b) {
        return migraphx::float_equal(a, b);
    }));
}
