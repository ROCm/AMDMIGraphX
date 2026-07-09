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
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
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

// Zip the stringized enumerator names (split on commas, dropping any `= value` suffix) with
// their captured values into a lookup table.
template <class Enum>
std::vector<std::pair<std::string, Enum>> make_enum_entries(const std::string& names,
                                                            std::initializer_list<Enum> values)
{
    auto parts = split_string(names, ',');
    if(parts.size() != values.size())
        MIGRAPHX_THROW("MIGRAPHX_ENUM: number of names does not match number of values");
    std::vector<std::pair<std::string, Enum>> entries;
    entries.reserve(values.size());
    std::transform(parts.begin(),
                   parts.end(),
                   values.begin(),
                   std::back_inserter(entries),
                   [](const std::string& part, Enum value) {
                       auto pos  = part.find('=');
                       auto text = pos == std::string::npos ? part : part.substr(0, pos);
                       return std::make_pair(trim(text), value);
                   });
    return entries;
}

template <class Enum>
std::string enum_to_string(Enum value)
{
    const auto& entries = migraphx_enum_entries(value);
    auto it             = std::find_if(
        entries.begin(), entries.end(), [&](const auto& p) { return p.second == value; });
    if(it == entries.end())
        MIGRAPHX_THROW("Invalid value for enum " + get_type_name<Enum>());
    return it->first;
}

} // namespace detail

// Returns the name/value table for an enum declared with MIGRAPHX_ENUM.
template <class Enum>
const std::vector<std::pair<std::string, Enum>>& enum_entries()
{
    static_assert(std::is_enum<Enum>{}, "enum_entries<Enum> requires an enum type");
    return migraphx_enum_entries(Enum{});
}

// Converts the name of an enumerator back into its value, throwing when the name is unknown.
template <class Enum>
Enum from_string(const std::string& name)
{
    static_assert(std::is_enum<Enum>{}, "from_string<Enum> requires an enum type");
    const auto& entries = migraphx_enum_entries(Enum{});
    auto it             = std::find_if(
        entries.begin(), entries.end(), [&](const auto& p) { return p.first == name; });
    if(it == entries.end())
        MIGRAPHX_THROW("Invalid name '" + name + "' for enum " + get_type_name<Enum>());
    return it->second;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Rewrites a single enumerator `x` (which may include an `= value`) so that operator->* captures
// its value. See the note on enum_capture above for why operator->* is used here.
#define MIGRAPHX_ENUM_PP_CAPTURE(x) (migraphx::detail::enum_capturer{}->*x)

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
// Supports explicit enumerator values and up to 16 enumerators. When used in a .cpp rather than a
// header, place it in an anonymous namespace so the generated helpers get internal linkage.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_ENUM(name, ...)                                                        \
    enum name                                                                           \
    {                                                                                   \
        __VA_ARGS__                                                                     \
    };                                                                                  \
    inline const std::vector<std::pair<std::string, name>>& migraphx_enum_entries(name) \
    {                                                                                   \
        static const std::vector<std::pair<std::string, name>> entries =                \
            migraphx::detail::make_enum_entries<name>(                                  \
                #__VA_ARGS__,                                                           \
                {MIGRAPHX_PP_TRANSFORM_ARGS(MIGRAPHX_ENUM_PP_CAPTURE, __VA_ARGS__)});   \
        return entries;                                                                 \
    }                                                                                   \
    inline std::string to_string(name value) { return migraphx::detail::enum_to_string(value); }

#endif // MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP
