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
#include <migraphx/enum.hpp>
#include <algorithm>
#include "test.hpp"

MIGRAPHX_ENUM(color, red, green = 5, blue)

// The example from the feature request: an explicit value on the last enumerator.
MIGRAPHX_ENUM(my_enum, first, last = 10)

// Single enumerator, exercises the base case of the value-capturing expansion.
MIGRAPHX_ENUM(solo, only_one)

// Expression-valued enumerators, including one that references a previous enumerator.
MIGRAPHX_ENUM(flags, none = 0, bit0 = 1 << 0, bit1 = 1 << 1, both = bit0 + bit1)

namespace migraphx {
// Declared inside the migraphx namespace to make sure the generated to_string(status) does not
// become ambiguous with the generic migraphx::to_string(const T&).
MIGRAPHX_ENUM(status, ok, busy = 4, done)
} // namespace migraphx

TEST_CASE(underlying_values)
{
    EXPECT(static_cast<int>(red) == 0);
    EXPECT(static_cast<int>(green) == 5);
    // Value continues incrementing after an explicit enumerator.
    EXPECT(static_cast<int>(blue) == 6);
}

TEST_CASE(to_string_names)
{
    EXPECT(to_string(red) == "red");
    EXPECT(to_string(green) == "green");
    EXPECT(to_string(blue) == "blue");
    EXPECT(to_string(only_one) == "only_one");
}

TEST_CASE(from_string_values)
{
    EXPECT(migraphx::from_string<color>("red") == red);
    EXPECT(migraphx::from_string<color>("green") == green);
    EXPECT(migraphx::from_string<color>("blue") == blue);
}

TEST_CASE(explicit_last_value)
{
    EXPECT(static_cast<int>(first) == 0);
    EXPECT(static_cast<int>(last) == 10);
    EXPECT(to_string(first) == "first");
    EXPECT(to_string(last) == "last");
    EXPECT(migraphx::from_string<my_enum>("last") == last);
}

TEST_CASE(expression_values)
{
    EXPECT(static_cast<int>(both) == 3);
    EXPECT(to_string(both) == "both");
    EXPECT(migraphx::from_string<flags>("bit1") == bit1);
}

TEST_CASE(entries_table)
{
    auto entries = migraphx::enum_entries<color>();
    EXPECT(entries.size() == 3);
    EXPECT(entries[0] == std::make_pair(std::string("red"), red));
    EXPECT(entries[1] == std::make_pair(std::string("green"), green));
    EXPECT(entries[2] == std::make_pair(std::string("blue"), blue));
}

TEST_CASE(round_trip)
{
    auto entries = migraphx::enum_entries<color>();
    EXPECT(std::all_of(entries.begin(), entries.end(), [](const auto& p) {
        return migraphx::from_string<color>(p.first) == p.second and to_string(p.second) == p.first;
    }));
}

TEST_CASE(from_string_unknown_throws)
{
    EXPECT(test::throws([] { migraphx::from_string<color>("purple"); }));
}

TEST_CASE(to_string_unknown_throws)
{
    EXPECT(test::throws([] { to_string(static_cast<color>(999)); }));
}

TEST_CASE(namespace_scoped_enum)
{
    // Both the ADL-found overload and the qualified call resolve to the generated overload.
    EXPECT(to_string(migraphx::ok) == "ok");
    EXPECT(migraphx::to_string(migraphx::busy) == "busy");
    EXPECT(static_cast<int>(migraphx::done) == 5);
    EXPECT(migraphx::from_string<migraphx::status>("done") == migraphx::done);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
