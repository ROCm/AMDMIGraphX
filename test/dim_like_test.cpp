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

#include <migraphx/dim_like.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>

#include <algorithm>
#include <sstream>
#include <vector>

#include "test.hpp"

using migraphx::dim_like;
using dd = migraphx::shape::dynamic_dimension;

static dim_like round_trip(const dim_like& d)
{
    auto v = migraphx::to_value(d);
    return migraphx::from_value<dim_like>(v);
}

// ===================================================================
// Construction and alternative inspection
// ===================================================================

TEST_CASE(construct_default)
{
    dim_like d;
    EXPECT(std::holds_alternative<int64_t>(d.value));
    EXPECT(std::get<int64_t>(d.value) == 0);
}

TEST_CASE(construct_int_marker_zero)
{
    dim_like d = 0;
    EXPECT(std::holds_alternative<int64_t>(d.value));
    EXPECT(std::get<int64_t>(d.value) == 0);
}

TEST_CASE(construct_int_marker_neg_one)
{
    dim_like d = -1;
    EXPECT(std::holds_alternative<int64_t>(d.value));
    EXPECT(std::get<int64_t>(d.value) == -1);
}

TEST_CASE(construct_int_value)
{
    dim_like d = 42;
    EXPECT(std::holds_alternative<int64_t>(d.value));
    EXPECT(std::get<int64_t>(d.value) == 42);
}

TEST_CASE(construct_from_size_t)
{
    std::size_t n = 7;
    dim_like d    = n;
    EXPECT(std::holds_alternative<int64_t>(d.value));
    EXPECT(std::get<int64_t>(d.value) == 7);
}

TEST_CASE(construct_from_dynamic_dimension_range)
{
    dim_like d = dd{1, 4};
    EXPECT(std::holds_alternative<dd>(d.value));
    EXPECT(std::get<dd>(d.value) == dd{1, 4});
}

TEST_CASE(construct_from_dynamic_dimension_symbolic)
{
    dim_like d = dd{migraphx::sym::var("n", {1, 8})};
    EXPECT(std::holds_alternative<dd>(d.value));
    EXPECT(std::get<dd>(d.value).is_symbolic());
}

// ===================================================================
// Equality / count semantics for legacy 0/-1 marker patterns
// ===================================================================

TEST_CASE(equality_int_marker_zero)
{
    dim_like d = 0;
    EXPECT(d == 0);
    EXPECT(0 == d);
    EXPECT(not(d == -1));
}

TEST_CASE(equality_int_marker_neg_one)
{
    dim_like d = -1;
    EXPECT(d == -1);
    EXPECT(-1 == d);
    EXPECT(not(d == 0));
}

TEST_CASE(equality_int_value)
{
    dim_like d = 5;
    EXPECT(d == 5);
    EXPECT(5 == d);
    EXPECT(d != 4);
}

TEST_CASE(equality_dd_alternative_never_matches_marker)
{
    dim_like d = dd{0, 4};
    EXPECT(d != 0);
    EXPECT(d != -1);
}

TEST_CASE(equality_between_alternatives)
{
    dim_like a = 3;
    dim_like b = dd{3, 3};
    EXPECT(a != b);
}

TEST_CASE(adl_get_and_holds_alternative)
{
    using migraphx::get;
    using migraphx::holds_alternative;

    dim_like d_int = 42;
    EXPECT(holds_alternative<int64_t>(d_int));
    EXPECT(get<int64_t>(d_int) == 42);

    dim_like d_dd = dd{1, 4};
    EXPECT(holds_alternative<dd>(d_dd));
    EXPECT(get<dd>(d_dd) == dd{1, 4});

    EXPECT(test::throws([&] { (void)get<dd>(d_int); }));
}

TEST_CASE(std_count_marker)
{
    std::vector<dim_like> dims = {0, 0, 6, -1};
    EXPECT(std::count(dims.begin(), dims.end(), -1) == 1);
    EXPECT(std::count(dims.begin(), dims.end(), 0) == 2);
}

// ===================================================================
// Streaming
// ===================================================================

TEST_CASE(stream_int)
{
    std::ostringstream ss;
    ss << dim_like{42};
    EXPECT(ss.str() == "42");
}

TEST_CASE(stream_neg_one)
{
    std::ostringstream ss;
    ss << dim_like{-1};
    EXPECT(ss.str() == "-1");
}

TEST_CASE(stream_dd)
{
    std::ostringstream ss;
    ss << dim_like{dd{1, 4}};
    std::ostringstream expected;
    expected << dd{1, 4};
    EXPECT(ss.str() == expected.str());
}

TEST_CASE(stream_dd_symbolic)
{
    auto sd = dd{migraphx::sym::var("n", {1, 8})};
    std::ostringstream ss;
    ss << dim_like{sd};
    std::ostringstream expected;
    expected << sd;
    EXPECT(ss.str() == expected.str());
}

// ===================================================================
// Serialization round-trip
// ===================================================================

TEST_CASE(serialize_int_zero)
{
    dim_like d = 0;
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<int64_t>(rt.value));
}

TEST_CASE(serialize_int_neg_one)
{
    dim_like d = -1;
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<int64_t>(rt.value));
}

TEST_CASE(serialize_int_value)
{
    dim_like d = 42;
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<int64_t>(rt.value));
}

TEST_CASE(serialize_dd_range)
{
    dim_like d = dd{1, 4};
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<dd>(rt.value));
}

TEST_CASE(serialize_dd_symbolic)
{
    dim_like d = dd{migraphx::sym::var("n", {1, 8})};
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<dd>(rt.value));
}

// ===================================================================
// Backward-compat: legacy serialized models stored dims as a plain int64
// array. Decoding such a value into vector<dim_like> must succeed and
// produce the int alternative for every entry.
// ===================================================================

TEST_CASE(from_value_legacy_int_array)
{
    std::vector<int64_t> legacy = {0, 0, 6, -1};
    auto loaded = migraphx::from_value<std::vector<dim_like>>(migraphx::to_value(legacy));
    EXPECT(loaded == std::vector<dim_like>{0, 0, 6, -1});
}

TEST_CASE(from_value_size_t_array)
{
    // Common path at op-construction sites: make_op("...", {{"dims", lens()}})
    // where lens() is vector<size_t>. The value layer routes that through uint64.
    std::vector<std::size_t> lens = {4, 24, 1};
    auto loaded = migraphx::from_value<std::vector<dim_like>>(migraphx::to_value(lens));
    EXPECT(loaded == std::vector<dim_like>{4, 24, 1});
}

TEST_CASE(to_value_int_array_byte_compat)
{
    // A vector<dim_like> holding only int alternatives must serialize to the
    // same value as the equivalent vector<int64_t>, so models with no symbolic
    // dims save byte-identical to today.
    std::vector<dim_like> dims = {0, 0, 6, -1};
    std::vector<int64_t> legacy{0, 0, 6, -1};
    EXPECT(migraphx::to_value(dims) == migraphx::to_value(legacy));
}

TEST_CASE(round_trip_mixed_vector)
{
    std::vector<dim_like> dims = {
        dim_like{0},
        dim_like{42},
        dim_like{dd{1, 4}},
        dim_like{-1},
    };
    auto v      = migraphx::to_value(dims);
    auto loaded = migraphx::from_value<std::vector<dim_like>>(v);
    EXPECT(loaded == dims);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
