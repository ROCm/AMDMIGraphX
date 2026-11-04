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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_CHARCONV_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_CHARCONV_HPP

#include <migraphx/config.hpp>
#include <migraphx/ranges.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct to_chars_result
{
    char* ptr;
    std::errc ec;
};

namespace charconv_detail {

template <class T>
bool round_trips(const std::string& value, T original)
{
    std::istringstream ss{value};
    ss.imbue(std::locale::classic());
    T parsed{};
    ss >> parsed;
    return static_cast<bool>(ss) and not(parsed < original) and not(original < parsed) and
           std::signbit(parsed) == std::signbit(original);
}

template <class T>
std::pair<std::string, std::errc> format_floating_point(T value)
{
    if(std::isnan(value))
        return {std::signbit(value) ? "-nan" : "nan", {}};
    if(std::isinf(value))
        return {std::signbit(value) ? "-inf" : "inf", {}};

    std::string result;
    auto precisions = range(1, std::numeric_limits<T>::max_digits10 + 1);
    auto it         = std::find_if(precisions.begin(), precisions.end(), [&](auto precision) {
        std::ostringstream ss;
        ss.imbue(std::locale::classic());
        ss.precision(precision);
        ss << value;
        result = ss.str();
        return round_trips(result, value);
    });
    if(it == precisions.end())
        return {{}, std::errc::invalid_argument};
    return {result, {}};
}

template <class T>
std::pair<std::string, std::errc> format(T value)
{
    static_assert(std::is_arithmetic<T>{}, "to_chars requires an arithmetic type");
    if constexpr(std::is_integral<T>{})
        return {std::to_string(value), {}};
    else
        return format_floating_point(value);
}

} // namespace charconv_detail

template <class T>
to_chars_result to_chars(char* first, char* last, T value)
{
    auto formatted = charconv_detail::format(value);
    if(formatted.second != std::errc{})
        return {first, formatted.second};
    if(static_cast<std::size_t>(last - first) < formatted.first.size())
        return {last, std::errc::value_too_large};
    return {std::copy(formatted.first.begin(), formatted.first.end(), first), {}};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_CHARCONV_HPP
