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
#include <migraphx/stringutils.hpp>
#include <migraphx/type_name.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace detail {

// enum_capture and enum_capturer implement the value capturing used by MIGRAPHX_ENUM. Each
// enumerator `e` in the list is rewritten to `enum_capturer{}->*name::e`. Since operator->*
// binds more tightly than assignment, `enum_capturer{}->*name::e = 42` parses as
// `(enum_capturer{}->*name::e) = 42`: operator->* captures the real value of the enumerator
// (which already accounts for the `= 42`), and operator= simply swallows the initializer.
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

// Preprocessor machinery: apply a macro to each enumerator in the list. The names use the
// `_PP_` infix so they are recognized as pure preprocessor helpers by clang-tidy.

#define MIGRAPHX_ENUM_PP_ARG_N(_1,  \
                               _2,  \
                               _3,  \
                               _4,  \
                               _5,  \
                               _6,  \
                               _7,  \
                               _8,  \
                               _9,  \
                               _10, \
                               _11, \
                               _12, \
                               _13, \
                               _14, \
                               _15, \
                               _16, \
                               _17, \
                               _18, \
                               _19, \
                               _20, \
                               _21, \
                               _22, \
                               _23, \
                               _24, \
                               _25, \
                               _26, \
                               _27, \
                               _28, \
                               _29, \
                               _30, \
                               _31, \
                               _32, \
                               N,   \
                               ...) \
    N
#define MIGRAPHX_ENUM_PP_RSEQ()                                                                    \
    32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, \
        8, 7, 6, 5, 4, 3, 2, 1, 0
#define MIGRAPHX_ENUM_PP_NARG(...) MIGRAPHX_ENUM_PP_NARG_IMPL(__VA_ARGS__, MIGRAPHX_ENUM_PP_RSEQ())
#define MIGRAPHX_ENUM_PP_NARG_IMPL(...) MIGRAPHX_ENUM_PP_ARG_N(__VA_ARGS__)

#define MIGRAPHX_ENUM_PP_CONCAT(a, b) MIGRAPHX_ENUM_PP_CONCAT_IMPL(a, b)
#define MIGRAPHX_ENUM_PP_CONCAT_IMPL(a, b) a##b

#define MIGRAPHX_ENUM_PP_CAPTURE(name, x) (migraphx::detail::enum_capturer{}->*name::x)

#define MIGRAPHX_ENUM_PP_EACH(m, name, ...)                                             \
    MIGRAPHX_ENUM_PP_CONCAT(MIGRAPHX_ENUM_PP_EACH_, MIGRAPHX_ENUM_PP_NARG(__VA_ARGS__)) \
    (m, name, __VA_ARGS__)

#define MIGRAPHX_ENUM_PP_EACH_1(m, name, x) m(name, x)
#define MIGRAPHX_ENUM_PP_EACH_2(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_1(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_3(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_2(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_4(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_3(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_5(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_4(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_6(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_5(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_7(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_6(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_8(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_7(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_9(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_8(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_10(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_9(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_11(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_10(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_12(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_11(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_13(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_12(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_14(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_13(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_15(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_14(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_16(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_15(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_17(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_16(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_18(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_17(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_19(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_18(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_20(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_19(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_21(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_20(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_22(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_21(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_23(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_22(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_24(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_23(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_25(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_24(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_26(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_25(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_27(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_26(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_28(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_27(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_29(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_28(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_30(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_29(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_31(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_30(m, name, __VA_ARGS__)
#define MIGRAPHX_ENUM_PP_EACH_32(m, name, x, ...) \
    m(name, x), MIGRAPHX_ENUM_PP_EACH_31(m, name, __VA_ARGS__)

// Declares an unscoped enum together with `to_string(name)` and `migraphx::from_string<name>`
// helpers for converting the enumerators to and from their names. Use it at namespace scope:
//
//     MIGRAPHX_ENUM(color,
//         red,
//         green = 5,
//         blue)
//
//     std::string s = to_string(green);              // "green"
//     color c       = migraphx::from_string<color>("blue"); // blue
//
// Supports explicit enumerator values and up to 32 enumerators.
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
                {MIGRAPHX_ENUM_PP_EACH(MIGRAPHX_ENUM_PP_CAPTURE, name, __VA_ARGS__)});  \
        return entries;                                                                 \
    }                                                                                   \
    inline std::string to_string(name value) { return migraphx::detail::enum_to_string(value); }

#endif // MIGRAPHX_GUARD_MIGRAPHX_ENUM_HPP
