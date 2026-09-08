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
    EXPECT(std::holds_alternative<int64_t>(d));
    EXPECT(std::get<int64_t>(d) == 0);
}

TEST_CASE(construct_int_marker_zero)
{
    dim_like d = 0;
    EXPECT(std::holds_alternative<int64_t>(d));
    EXPECT(std::get<int64_t>(d) == 0);
}

TEST_CASE(construct_int_marker_neg_one)
{
    dim_like d = -1;
    EXPECT(std::holds_alternative<int64_t>(d));
    EXPECT(std::get<int64_t>(d) == -1);
}

TEST_CASE(construct_int_value)
{
    dim_like d = 42;
    EXPECT(std::holds_alternative<int64_t>(d));
    EXPECT(std::get<int64_t>(d) == 42);
}

TEST_CASE(construct_from_size_t)
{
    std::size_t n = 7;
    dim_like d    = n;
    EXPECT(std::holds_alternative<int64_t>(d));
    EXPECT(std::get<int64_t>(d) == 7);
}

TEST_CASE(construct_from_dynamic_dimension_range)
{
    dim_like d = dd{1, 4};
    EXPECT(std::holds_alternative<dd>(d));
    EXPECT(std::get<dd>(d) == dd{1, 4});
}

TEST_CASE(construct_from_dynamic_dimension_symbolic)
{
    dim_like d = dd{migraphx::sym::var("n", {1, 8})};
    EXPECT(std::holds_alternative<dd>(d));
    EXPECT(std::get<dd>(d).is_symbolic());
}

TEST_CASE(get_throws_on_wrong_alternative)
{
    dim_like d = 42;
    EXPECT(test::throws([&] { (void)std::get<dd>(d); }));
}

// ===================================================================
// Equality and std::count on 0 / -1 markers
// ===================================================================

TEST_CASE(equality_int_marker_zero)
{
    dim_like d = 0;
    EXPECT(d == dim_like{0});
    EXPECT(dim_like{0} == d);
    EXPECT(not(d == dim_like{-1}));
}

TEST_CASE(equality_int_marker_neg_one)
{
    dim_like d = -1;
    EXPECT(d == dim_like{-1});
    EXPECT(dim_like{-1} == d);
    EXPECT(not(d == dim_like{0}));
}

TEST_CASE(equality_int_value)
{
    dim_like d = 5;
    EXPECT(d == dim_like{5});
    EXPECT(dim_like{5} == d);
    EXPECT(d != dim_like{4});
}

TEST_CASE(equality_dd_alternative_never_matches_marker)
{
    dim_like d = dd{0, 4};
    EXPECT(d != dim_like{0});
    EXPECT(d != dim_like{-1});
}

TEST_CASE(equality_between_alternatives)
{
    dim_like a = 3;
    dim_like b = dd{3, 3};
    EXPECT(a != b);
}

TEST_CASE(std_count_marker)
{
    std::vector<dim_like> dims = {0, 0, 6, -1};
    EXPECT(std::count(dims.begin(), dims.end(), dim_like{-1}) == 1);
    EXPECT(std::count(dims.begin(), dims.end(), dim_like{0}) == 2);
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
    EXPECT(std::holds_alternative<int64_t>(rt));
}

TEST_CASE(serialize_int_neg_one)
{
    dim_like d = -1;
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<int64_t>(rt));
}

TEST_CASE(serialize_int_value)
{
    dim_like d = 42;
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<int64_t>(rt));
}

TEST_CASE(serialize_dd_range)
{
    dim_like d = dd{1, 4};
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<dd>(rt));
}

TEST_CASE(serialize_dd_symbolic)
{
    dim_like d = dd{migraphx::sym::var("n", {1, 8})};
    auto rt    = round_trip(d);
    EXPECT(rt == d);
    EXPECT(std::holds_alternative<dd>(rt));
}

// ===================================================================
// Backward-compat: load and save against legacy int / size_t arrays
// ===================================================================

TEST_CASE(from_value_legacy_int_array)
{
    std::vector<int64_t> legacy = {0, 0, 6, -1};
    auto loaded = migraphx::from_value<std::vector<dim_like>>(migraphx::to_value(legacy));
    EXPECT(loaded == std::vector<dim_like>{0, 0, 6, -1});
}

TEST_CASE(from_value_size_t_array)
{
    std::vector<std::size_t> lens = {4, 24, 1};
    auto loaded = migraphx::from_value<std::vector<dim_like>>(migraphx::to_value(lens));
    EXPECT(loaded == std::vector<dim_like>{4, 24, 1});
}

TEST_CASE(to_value_int_array_byte_compat)
{
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

// ===================================================================
// ADL visit
// ===================================================================

TEST_CASE(visit_int)
{
    dim_like d = 42;
    auto seen  = visit(
        [](const auto& x) -> std::string {
            if constexpr(std::is_same<std::decay_t<decltype(x)>, int64_t>{})
                return "int";
            else
                return "dd";
        },
        d);
    EXPECT(seen == "int");
}

TEST_CASE(visit_dd)
{
    dim_like d = dd{1, 4};
    auto seen  = visit(
        [](const auto& x) -> std::string {
            if constexpr(std::is_same<std::decay_t<decltype(x)>, int64_t>{})
                return "int";
            else
                return "dd";
        },
        d);
    EXPECT(seen == "dd");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
