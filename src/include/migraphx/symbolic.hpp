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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_SYMBOLIC_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_SYMBOLIC_HPP

#include <cstddef>
#include <map>
#include <memory>
#include <ostream>
#include <string>

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

struct MIGRAPHX_EXPORT symbolic_expr
{
    symbolic_expr();
    explicit symbolic_expr(std::size_t n);
    explicit symbolic_expr(const std::string& s);

    bool empty() const;
    std::string to_string() const;
    std::size_t eval(const std::map<std::string, std::size_t>& symbol_map) const;

    MIGRAPHX_EXPORT friend symbolic_expr operator+(const symbolic_expr& a,
                                                   const symbolic_expr& b);
    MIGRAPHX_EXPORT friend symbolic_expr operator-(const symbolic_expr& a,
                                                   const symbolic_expr& b);
    MIGRAPHX_EXPORT friend symbolic_expr operator*(const symbolic_expr& a,
                                                   const symbolic_expr& b);
    MIGRAPHX_EXPORT friend symbolic_expr operator/(const symbolic_expr& a,
                                                   const symbolic_expr& b);
    MIGRAPHX_EXPORT friend bool operator==(const symbolic_expr& a, const symbolic_expr& b);
    MIGRAPHX_EXPORT friend bool operator!=(const symbolic_expr& a, const symbolic_expr& b);
    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const symbolic_expr& e);

    struct impl;

    private:
    symbolic_expr(std::shared_ptr<const impl> pi);
    std::shared_ptr<const impl> p;
};

inline symbolic_expr operator+(const symbolic_expr& a, std::size_t b) { return a + symbolic_expr(b); }
inline symbolic_expr operator+(std::size_t a, const symbolic_expr& b) { return symbolic_expr(a) + b; }
inline symbolic_expr operator-(const symbolic_expr& a, std::size_t b) { return a - symbolic_expr(b); }
inline symbolic_expr operator-(std::size_t a, const symbolic_expr& b) { return symbolic_expr(a) - b; }
inline symbolic_expr operator*(const symbolic_expr& a, std::size_t b) { return a * symbolic_expr(b); }
inline symbolic_expr operator*(std::size_t a, const symbolic_expr& b) { return symbolic_expr(a) * b; }
inline symbolic_expr operator/(const symbolic_expr& a, std::size_t b) { return a / symbolic_expr(b); }
inline symbolic_expr operator/(std::size_t a, const symbolic_expr& b) { return symbolic_expr(a) / b; }

MIGRAPHX_EXPORT void migraphx_to_value(value& v, const symbolic_expr& e);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, symbolic_expr& e);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
