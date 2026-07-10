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
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <string>
#include <type_traits>
#include "test.hpp"

// These enums are test-local fixtures; the anonymous namespace gives the macro-generated helper
// functions internal linkage.
namespace {
MIGRAPHX_ENUM(color, red, green = 5, blue)

// The example from the feature request: an explicit value on the last enumerator.
MIGRAPHX_ENUM(my_enum, first, last = 10)

// Single enumerator, exercises the base case of the value-capturing expansion.
MIGRAPHX_ENUM(solo, only_one)

// Expression-valued enumerators, including one that references a previous enumerator.
MIGRAPHX_ENUM(flags, none = 0, bit0 = 1, bit1 = 2, both = bit0 + bit1)

// Scoped enum (enum class) with an explicit value.
MIGRAPHX_ENUM_CLASS(scoped_color, cyan, magenta = 7, yellow)

// Enums nested in a struct, exercising the friend-based (MIGRAPHX_NESTED_*) variants.
struct gadget
{
    MIGRAPHX_NESTED_ENUM(mode, off, on = 3, standby)
    MIGRAPHX_NESTED_ENUM_CLASS(unit, mm, cm = 10, m)
};

// A plain enum declared without the macro, for the negative is_named_enum case.
enum plain_enum
{
    plain_a,
    plain_b
};

// A reflectable struct with named-enum fields, for the serialize round-trip.
struct config
{
    MIGRAPHX_NESTED_ENUM(mode, off, on = 3, standby)
    MIGRAPHX_NESTED_ENUM_CLASS(unit, mm, cm = 10, m)
    mode active_mode = on;
    unit length_unit = unit::cm;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.active_mode, "active_mode"),
                              f(self.length_unit, "length_unit"));
    }
};

// Sixty-three enumerators, the maximum supported by the underlying pp.hpp transform.
MIGRAPHX_ENUM(many,
              m0,
              m1,
              m2,
              m3,
              m4,
              m5,
              m6,
              m7,
              m8,
              m9,
              m10,
              m11,
              m12,
              m13,
              m14,
              m15,
              m16,
              m17,
              m18,
              m19,
              m20,
              m21,
              m22,
              m23,
              m24,
              m25,
              m26,
              m27,
              m28,
              m29,
              m30,
              m31,
              m32,
              m33,
              m34,
              m35,
              m36,
              m37,
              m38,
              m39,
              m40,
              m41,
              m42,
              m43,
              m44,
              m45,
              m46,
              m47,
              m48,
              m49,
              m50,
              m51,
              m52,
              m53,
              m54,
              m55,
              m56,
              m57,
              m58,
              m59,
              m60,
              m61,
              m62)
} // namespace

namespace migraphx {
// Declared with external linkage inside the migraphx namespace (as a real header would) so we can
// check that the generated to_string(status) is not ambiguous with migraphx::to_string(const T&).
// NOLINTNEXTLINE(misc-use-internal-linkage)
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
    EXPECT(entries[0] == red);
    EXPECT(entries[1] == green);
    EXPECT(entries[2] == blue);
}

TEST_CASE(round_trip)
{
    auto entries = migraphx::enum_entries<color>();
    EXPECT(std::all_of(entries.begin(), entries.end(), [](color value) {
        return migraphx::from_string<color>(to_string(value)) == value;
    }));
}

TEST_CASE(max_enumerators)
{
    auto entries = migraphx::enum_entries<many>();
    EXPECT(entries.size() == 63);
    EXPECT(entries.front() == m0);
    EXPECT(entries.back() == m62);
    EXPECT(static_cast<int>(m62) == 62);
    EXPECT(to_string(m32) == "m32");
    EXPECT(migraphx::from_string<many>("m47") == m47);
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

TEST_CASE(enum_class_values_and_strings)
{
    EXPECT(static_cast<int>(scoped_color::cyan) == 0);
    EXPECT(static_cast<int>(scoped_color::magenta) == 7);
    // Value continues incrementing after an explicit enumerator.
    EXPECT(static_cast<int>(scoped_color::yellow) == 8);
    EXPECT(to_string(scoped_color::cyan) == "cyan");
    EXPECT(to_string(scoped_color::magenta) == "magenta");
    EXPECT(migraphx::from_string<scoped_color>("yellow") == scoped_color::yellow);
}

TEST_CASE(enum_class_entries)
{
    auto entries = migraphx::enum_entries<scoped_color>();
    EXPECT(entries.size() == 3);
    EXPECT(entries[0] == scoped_color::cyan);
    EXPECT(entries[1] == scoped_color::magenta);
    EXPECT(entries[2] == scoped_color::yellow);
}

TEST_CASE(enum_class_is_scoped)
{
    // A real scoped enum is not implicitly convertible to its underlying type.
    EXPECT(not std::is_convertible<scoped_color, int>{});
}

TEST_CASE(enum_class_unknown_throws)
{
    EXPECT(test::throws([] { migraphx::from_string<scoped_color>("black"); }));
    EXPECT(test::throws([] { to_string(static_cast<scoped_color>(999)); }));
}

TEST_CASE(nested_enum)
{
    EXPECT(static_cast<int>(gadget::off) == 0);
    EXPECT(static_cast<int>(gadget::on) == 3);
    EXPECT(static_cast<int>(gadget::standby) == 4);
    EXPECT(to_string(gadget::on) == "on");
    EXPECT(migraphx::from_string<gadget::mode>("standby") == gadget::standby);
    auto entries = migraphx::enum_entries<gadget::mode>();
    EXPECT(entries.size() == 3);
    EXPECT(entries[0] == gadget::off);
    EXPECT(entries[2] == gadget::standby);
}

TEST_CASE(nested_enum_class)
{
    EXPECT(not std::is_convertible<gadget::unit, int>{});
    EXPECT(static_cast<int>(gadget::unit::mm) == 0);
    EXPECT(static_cast<int>(gadget::unit::cm) == 10);
    EXPECT(static_cast<int>(gadget::unit::m) == 11);
    EXPECT(to_string(gadget::unit::cm) == "cm");
    EXPECT(migraphx::from_string<gadget::unit>("m") == gadget::unit::m);
    EXPECT(test::throws([] { migraphx::from_string<gadget::unit>("km"); }));
}

TEST_CASE(is_named_enum_trait)
{
    // True for every MIGRAPHX_ENUM variant, including nested ones.
    EXPECT(migraphx::is_named_enum<color>{});
    EXPECT(migraphx::is_named_enum<scoped_color>{});
    EXPECT(migraphx::is_named_enum<gadget::mode>{});
    EXPECT(migraphx::is_named_enum<gadget::unit>{});
    // False for plain enums and non-enum types.
    EXPECT(not migraphx::is_named_enum<plain_enum>{});
    EXPECT(not migraphx::is_named_enum<int>{});
    EXPECT(not migraphx::is_named_enum<std::string>{});
}

TEST_CASE(value_stores_named_enum_as_string)
{
    migraphx::value v = green;
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "green");
    EXPECT(v.to<color>() == green);
}

TEST_CASE(value_retrieves_named_enum_from_string)
{
    migraphx::value v = "blue";
    EXPECT(v.to<color>() == blue);
}

TEST_CASE(value_retrieves_named_enum_from_int)
{
    migraphx::value v = 5;
    EXPECT(v.to<color>() == green);
}

TEST_CASE(value_assign_named_enum)
{
    migraphx::value v;
    v = green;
    EXPECT(v.is_string() and v.get_string() == "green");
}

TEST_CASE(value_keyed_named_enum)
{
    migraphx::value v("c", blue);
    EXPECT(v.get_key() == "c");
    EXPECT(v.is_string() and v.get_string() == "blue");
    EXPECT(v.to<color>() == blue);
}

TEST_CASE(value_scoped_named_enum_round_trip)
{
    migraphx::value v = scoped_color::magenta;
    EXPECT(v.is_string() and v.get_string() == "magenta");
    EXPECT(v.to<scoped_color>() == scoped_color::magenta);
    migraphx::value vi = 7;
    EXPECT(vi.to<scoped_color>() == scoped_color::magenta);
}

TEST_CASE(serialize_standalone_named_enum)
{
    // to_value stores the enum as its name.
    EXPECT(migraphx::to_value(green).is_string());
    EXPECT(migraphx::to_value(green).get_string() == "green");
    // from_value recovers it from the name string...
    EXPECT(migraphx::from_value<color>(migraphx::to_value(green)) == green);
    // ...and from an integer value.
    migraphx::value vi = 5;
    EXPECT(migraphx::from_value<color>(vi) == green);
}

TEST_CASE(serialize_named_enum_field)
{
    config c;
    c.active_mode = config::standby;
    c.length_unit = config::unit::m;

    migraphx::value v = migraphx::to_value(c);
    // The named-enum fields are serialized as their names.
    EXPECT(v.at("active_mode").is_string());
    EXPECT(v.at("active_mode").get_string() == "standby");
    EXPECT(v.at("length_unit").get_string() == "m");

    // And they round-trip back.
    auto c2 = migraphx::from_value<config>(v);
    EXPECT(c2.active_mode == config::standby);
    EXPECT(c2.length_unit == config::unit::m);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
