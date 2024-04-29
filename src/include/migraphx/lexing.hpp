/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_LEXING_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_LEXING_HPP

#include <functional>
#include <string>
#include <vector>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using lexer = std::function<const char*(const char* start, const char* end)>;

template <class P>
inline auto lex_while(P p)
{
    return [=](const char* start, const char* end) {
        return std::find_if(start, end, [&](char c) { return not p(c); });
    };
}

template <class P>
inline auto lex_if(P p)
{
    return [=](const char* start, const char*) {
        if(p(*start))
            return start + 1;
        return start;
    };
}

MIGRAPHX_EXPORT std::function<const char*(const char*, const char*)> lex_equal(const std::string&);

MIGRAPHX_EXPORT std::vector<std::string_view>
tokenize(const char*, const char*, const std::vector<lexer>&);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
