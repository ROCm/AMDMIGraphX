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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP

#include <algorithm>
#include <array>
#include <iterator>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <migraphx/array.hpp>
#include <migraphx/config.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/pp.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/type_name.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace detail {

// enum_capture and enum_capturer implement the value capturing used by MIGRAPHX_ENUM. Each
// enumerator `e` in the list is rewritten to `enum_capturer{}->*e`. Since operator->* binds
// more tightly than assignment, `enum_capturer{}->*e = 42` parses as
// `(enum_capturer{}->*e) = 42`: operator->* captures the real value of the enumerator (which
// already accounts for the `= 42`), and operator= simply swallows the initializer.
template <class T>
struct enum_capture
{
    T value;

    template <class U>
    constexpr enum_capture& operator=(U)
    {
        return *this;
    }

    constexpr operator T() const { return value; }
};

struct enum_capturer
{
    template <class T>
    constexpr enum_capture<T> operator->*(T value) const
    {
        return {value};
    }
};

// Wraps an enumerator as a type so that get_type_name renders it as its name: the compiler spells
// a named enumerator as its own identifier, e.g. get_type_name<enum_value<color, green>>() is
// "...enum_value<color, green>".
template <class T, T X>
struct enum_value
{
};

// Recovers the enumerator name from that type name, e.g. "green" from "...enum_value<color, green>"
// or "on" from "...enum_value<mode, mode::on>" (scoped and nested enumerators are qualified).
template <class T, T X>
std::string enum_value_name()
{
    const std::string& full = get_type_name<enum_value<T, X>>();
    auto begin              = full.find(',', full.find('<')) + 1;
    auto name               = trim(full.substr(begin, full.rfind('>') - begin));
    auto scope              = name.rfind("::");
    return scope == std::string::npos ? name : name.substr(scope + 2);
}

template <class Enum, std::size_t N>
auto enum_value_names()
{
    return sequence_c<N>([](auto... is) {
        constexpr auto entries = migraphx_enum_entries(Enum{});
        return make_array<std::string>(enum_value_name<Enum, entries[is]>()...);
    });
}

// Maps an enumerator value to its name. The value set comes from migraphx_enum_entries, but the
// names are derived from the values through get_type_name rather than from any stored strings. The
// value -> name table is built once per enum so lookups are O(1).
template <class Enum>
std::string enum_to_string(Enum value)
{
    static const auto lookup = [] {
        constexpr auto entries = migraphx_enum_entries(Enum{});
        constexpr auto n       = entries.size();
        const auto names       = enum_value_names<Enum, n>();
        std::unordered_map<Enum, std::string> result;
        std::transform(entries.begin(),
                       entries.end(),
                       names.begin(),
                       std::inserter(result, result.end()),
                       [](Enum v, const std::string& n) { return std::make_pair(v, n); });
        return result;
    }();
    auto it = lookup.find(value);
    if(it == lookup.end())
        MIGRAPHX_THROW("Invalid value for enum " + get_type_name<Enum>());
    return it->second;
}

} // namespace detail

// Detects enums declared with the MIGRAPHX_ENUM family: those provide a migraphx_enum_entries hook
// and therefore support to_string and migraphx::from_string.
template <class T, class = void>
struct is_named_enum : std::false_type
{
};

template <class T>
struct is_named_enum<T, std::void_t<decltype(migraphx_enum_entries(std::declval<T>()))>>
    : std::true_type
{
};

// Returns the array of enumerator values for an enum declared with MIGRAPHX_ENUM.
template <class Enum, MIGRAPHX_REQUIRES(is_named_enum<Enum>{})>
auto enum_entries()
{
    return migraphx_enum_entries(Enum{});
}

// Converts the name of an enumerator back into its value, throwing when the name is unknown. The
// name -> value table is built once per enum from migraphx_enum_entries so lookups are O(1).
template <class Enum, MIGRAPHX_REQUIRES(is_named_enum<Enum>{})>
Enum from_string(const std::string& name)
{
    static const auto lookup = [] {
        const auto entries = migraphx_enum_entries(Enum{});
        std::unordered_map<std::string, Enum> result;
        std::transform(entries.begin(),
                       entries.end(),
                       std::inserter(result, result.end()),
                       [](Enum value) { return std::make_pair(to_string(value), value); });
        return result;
    }();
    auto it = lookup.find(name);
    if(it == lookup.end())
        MIGRAPHX_THROW("Invalid name '" + name + "' for enum " + get_type_name<Enum>());
    return it->second;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Rewrites a single enumerator `x` (which may include an `= value`) so that operator->* captures
// its value. See the note on enum_capture above for why operator->* is used here.
#define MIGRAPHX_DETAIL_ENUM_CAPTURE(x) (migraphx::detail::enum_capturer{}->*x)

// The scoped-enum variant qualifies the enumerator with `enum_scope`, a local alias for the enum
// type that MIGRAPHX_ENUM_CLASS declares in the entries function (scoped enumerators are not
// visible unqualified).
#define MIGRAPHX_DETAIL_ENUM_CLASS_CAPTURE(x) (migraphx::detail::enum_capturer{}->*enum_scope::x)

// Generates the ADL hooks (migraphx_enum_entries + to_string) shared by the MIGRAPHX_ENUM family.
// `linkage` is `inline` for a namespace-scope enum or `friend` for a class-scope (nested) enum;
// `capture` is the per-enumerator capture macro; `prologue` runs before the capture list and is
// used by the scoped variants to declare the enum_scope alias. migraphx_enum_entries returns just
// the array of enumerator values; to_string recovers the names from those values via get_type_name.
#ifdef CPPCHECK
// cppcheck's preprocessor cannot expand the recursive MIGRAPHX_PP_TRANSFORM_ARGS, so generate the
// hooks without it; the captured values are irrelevant to static analysis.
#define MIGRAPHX_DETAIL_ENUM_HELPERS(linkage, name, capture, prologue, ...) \
    linkage constexpr auto migraphx_enum_entries(name)                      \
    {                                                                       \
        return migraphx::make_array<name>(name{});                          \
    }                                                                       \
    linkage std::string to_string(name value) { return migraphx::detail::enum_to_string(value); }
#else
#define MIGRAPHX_DETAIL_ENUM_HELPERS(linkage, name, capture, prologue, ...) \
    linkage constexpr auto migraphx_enum_entries(name)                      \
    {                                                                       \
        prologue return migraphx::make_array<name>(                         \
            MIGRAPHX_PP_TRANSFORM_ARGS(capture, __VA_ARGS__));              \
    }                                                                       \
    linkage std::string to_string(name value) { return migraphx::detail::enum_to_string(value); }
#endif

// Declares an unscoped enum and generates `to_string` and `migraphx::from_string` helpers that
// convert its enumerators to and from their names. Use it at namespace scope:
//
//     MIGRAPHX_ENUM(color,
//         red,
//         green = 5,
//         blue)
//
//     std::string s = to_string(green);                     // "green"
//     color c       = migraphx::from_string<color>("blue"); // blue
//
// Enumerators may take explicit values, and up to 63 are supported. When used in a .cpp instead of
// a header, place it in an anonymous namespace so the generated helpers get internal linkage.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_ENUM(name, ...) \
    enum name                    \
    {                            \
        __VA_ARGS__              \
    };                           \
    MIGRAPHX_DETAIL_ENUM_HELPERS(inline, name, MIGRAPHX_DETAIL_ENUM_CAPTURE, , __VA_ARGS__)

// Like MIGRAPHX_ENUM, but declares a scoped enum (enum class):
//
//     MIGRAPHX_ENUM_CLASS(color,
//         red,
//         green = 5,
//         blue)
//
//     std::string s = to_string(color::green);              // "green"
//     color c       = migraphx::from_string<color>("blue"); // color::blue
//
// An explicit enumerator value must be self-contained: it cannot reference another enumerator,
// which is not visible unqualified in a scoped enum.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_ENUM_CLASS(name, ...) \
    enum class name                    \
    {                                  \
        __VA_ARGS__                    \
    };                                 \
    MIGRAPHX_DETAIL_ENUM_HELPERS(      \
        inline, name, MIGRAPHX_DETAIL_ENUM_CLASS_CAPTURE, using enum_scope = name;, __VA_ARGS__)

// Like MIGRAPHX_ENUM, but for an enum declared inside a class or struct. The helpers are generated
// as hidden friends instead of free functions so that argument-dependent lookup still finds them.
// Use it inside the class/struct body:
//
//     struct widget
//     {
//         MIGRAPHX_NESTED_ENUM(mode, off, on = 3, standby)
//     };
//
//     std::string s  = to_string(widget::on);                      // "on"
//     widget::mode m = migraphx::from_string<widget::mode>("off"); // widget::off
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_NESTED_ENUM(name, ...) \
    enum name                           \
    {                                   \
        __VA_ARGS__                     \
    };                                  \
    MIGRAPHX_DETAIL_ENUM_HELPERS(friend, name, MIGRAPHX_DETAIL_ENUM_CAPTURE, , __VA_ARGS__)

// Like MIGRAPHX_NESTED_ENUM, but declares a scoped enum (enum class); it relates to
// MIGRAPHX_NESTED_ENUM as MIGRAPHX_ENUM_CLASS does to MIGRAPHX_ENUM, including the same restriction
// on explicit enumerator values. Use it inside the class/struct body:
//
//     struct widget
//     {
//         MIGRAPHX_NESTED_ENUM_CLASS(unit, mm, cm = 10, m)
//     };
//
//     std::string s  = to_string(widget::unit::cm);               // "cm"
//     widget::unit u = migraphx::from_string<widget::unit>("mm"); // widget::unit::mm
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_NESTED_ENUM_CLASS(name, ...) \
    enum class name                           \
    {                                         \
        __VA_ARGS__                           \
    };                                        \
    MIGRAPHX_DETAIL_ENUM_HELPERS(             \
        friend, name, MIGRAPHX_DETAIL_ENUM_CLASS_CAPTURE, using enum_scope = name;, __VA_ARGS__)

#endif // MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP
