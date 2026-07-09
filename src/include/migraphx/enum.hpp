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
#include <migraphx/pp.hpp>
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

// Looks up the name of an enumerator. The entries array (from migraphx_enum_entries) holds only
// the values, so the name is the correspondingly-positioned token from the stringized enumerator
// list, with any `= value` suffix stripped.
template <class Enum>
std::string enum_to_string(const std::string& names, Enum value)
{
    const auto entries = migraphx_enum_entries(value);
    auto parts         = split_string(names, ',');
    if(parts.size() != entries.size())
        MIGRAPHX_THROW("MIGRAPHX_ENUM: too many enumerators for " + get_type_name<Enum>());
    const auto* it = std::find(entries.begin(), entries.end(), value);
    if(it == entries.end())
        MIGRAPHX_THROW("Invalid value for enum " + get_type_name<Enum>());
    const auto& part = parts[it - entries.begin()];
    auto pos         = part.find('=');
    return trim(pos == std::string::npos ? part : part.substr(0, pos));
}

} // namespace detail

// Returns the array of enumerator values for an enum declared with MIGRAPHX_ENUM.
template <class Enum>
auto enum_entries()
{
    static_assert(std::is_enum<Enum>{}, "enum_entries<Enum> requires an enum type");
    return migraphx_enum_entries(Enum{});
}

// Converts the name of an enumerator back into its value, throwing when the name is unknown. The
// name -> value table is built once per enum from migraphx_enum_entries so lookups are O(1).
template <class Enum>
Enum from_string(const std::string& name)
{
    static_assert(std::is_enum<Enum>{}, "from_string<Enum> requires an enum type");
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
#define MIGRAPHX_ENUM_PP_CAPTURE(x) (migraphx::detail::enum_capturer{}->*x)

// The scoped-enum variant qualifies the enumerator with `enum_scope`, a local alias for the enum
// type that MIGRAPHX_ENUM_CLASS declares in the entries function (scoped enumerators are not
// visible unqualified).
#define MIGRAPHX_ENUM_CLASS_PP_CAPTURE(x) (migraphx::detail::enum_capturer{}->*enum_scope::x)

// Declares an unscoped enum together with `to_string(name)` and `migraphx::from_string<name>`
// helpers for converting the enumerators to and from their names. Use it at namespace scope:
//
//     MIGRAPHX_ENUM(color,
//         red,
//         green = 5,
//         blue)
//
//     std::string s = to_string(green);                     // "green"
//     color c       = migraphx::from_string<color>("blue"); // blue
//
// migraphx_enum_entries returns just the array of enumerator values; the names are recovered on
// demand from the stringized enumerator list by to_string.
//
// Supports explicit enumerator values and up to 63 enumerators. When used in a .cpp rather than a
// header, place it in an anonymous namespace so the generated helpers get internal linkage.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_ENUM(name, ...)                                                \
    enum name                                                                   \
    {                                                                           \
        __VA_ARGS__                                                             \
    };                                                                          \
    inline auto migraphx_enum_entries(name)                                     \
    {                                                                           \
        return migraphx::make_array<name>(                                      \
            MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_ENUM_PP_CAPTURE, __VA_ARGS__)); \
    }                                                                           \
    inline std::string to_string(name value)                                    \
    {                                                                           \
        return migraphx::detail::enum_to_string(#__VA_ARGS__, value);           \
    }

// Like MIGRAPHX_ENUM, but declares a scoped enum (enum class). The enumerators are captured
// through a local `enum_scope` alias since they are not visible unqualified. to_string and
// migraphx::from_string work the same way:
//
//     MIGRAPHX_ENUM_CLASS(color,
//         red,
//         green = 5,
//         blue)
//
//     std::string s = to_string(color::green);              // "green"
//     color c       = migraphx::from_string<color>("blue"); // color::blue
//
// Explicit enumerator values must be self-contained (literals or expressions that do not reference
// other enumerators, which are not visible unqualified in a scoped enum). Supports up to 63
// enumerators. When used in a .cpp rather than a header, place it in an anonymous namespace so the
// generated helpers get internal linkage.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_ENUM_CLASS(name, ...)                                                \
    enum class name                                                                   \
    {                                                                                 \
        __VA_ARGS__                                                                   \
    };                                                                                \
    inline auto migraphx_enum_entries(name)                                           \
    {                                                                                 \
        using enum_scope = name;                                                      \
        return migraphx::make_array<name>(                                            \
            MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_ENUM_CLASS_PP_CAPTURE, __VA_ARGS__)); \
    }                                                                                 \
    inline std::string to_string(name value)                                          \
    {                                                                                 \
        return migraphx::detail::enum_to_string(#__VA_ARGS__, value);                 \
    }

#endif // MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP
