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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_SYM_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_SYM_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>

#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

namespace sym {

struct expr;
MIGRAPHX_EXPORT expr var(const std::string& name,
                         int64_t min                = 1,
                         int64_t max                = 1,
                         std::set<int64_t> optimals = {});
MIGRAPHX_EXPORT expr lit(int64_t n);
MIGRAPHX_EXPORT expr parse(const std::string& s);

struct MIGRAPHX_EXPORT expr
{
    expr();

    bool empty() const;
    std::size_t hash() const;
    std::string to_string() const;
    value to_value() const;
    void from_value(const value& v);
    std::size_t eval_uint(const std::unordered_map<expr, std::size_t>& symbol_map) const;
    std::pair<int64_t, int64_t> compute_bounds() const;
    std::pair<std::size_t, std::size_t> compute_bounds_uint() const;
    int64_t eval_min() const;
    int64_t eval_max() const;
    std::size_t eval_min_uint() const;
    std::size_t eval_max_uint() const;
    std::set<int64_t> eval_optimals() const;
    std::set<std::size_t> eval_optimals_uint() const;
    expr subs(const std::unordered_map<expr, expr>& symbol_map) const;

    MIGRAPHX_EXPORT friend expr operator+(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend expr operator-(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend expr operator*(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend expr operator/(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend bool operator==(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend bool operator!=(const expr& a, const expr& b);
    MIGRAPHX_EXPORT friend bool operator<(const expr& a, const expr& b);
    friend bool operator>(const expr& a, const expr& b) { return b < a; }
    friend bool operator<=(const expr& a, const expr& b) { return not(b < a); }
    friend bool operator>=(const expr& a, const expr& b) { return not(a < b); }
    MIGRAPHX_EXPORT friend std::ostream& operator<<(std::ostream& os, const expr& e);

    friend expr operator+(const expr& a, int64_t b) { return a + lit(b); }
    friend expr operator+(int64_t a, const expr& b) { return lit(a) + b; }
    friend expr operator-(const expr& a, int64_t b) { return a - lit(b); }
    friend expr operator-(int64_t a, const expr& b) { return lit(a) - b; }
    friend expr operator*(const expr& a, int64_t b) { return a * lit(b); }
    friend expr operator*(int64_t a, const expr& b) { return lit(a) * b; }
    friend expr operator/(const expr& a, int64_t b) { return a / lit(b); }
    friend expr operator/(int64_t a, const expr& b) { return lit(a) / b; }

    struct impl;

    MIGRAPHX_EXPORT friend expr
    var(const std::string& name, int64_t min, int64_t max, std::set<int64_t> optimals);
    MIGRAPHX_EXPORT friend expr lit(int64_t n);
    MIGRAPHX_EXPORT friend expr parse(const std::string& s);

    private:
    expr(std::shared_ptr<const impl> pi);
    std::shared_ptr<const impl> p;
};

} // namespace sym

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {
template <>
struct hash<migraphx::sym::expr>
{
    using argument_type = migraphx::sym::expr;
    using result_type   = std::size_t;
    result_type operator()(const migraphx::sym::expr& e) const { return e.hash(); }
};
} // namespace std

#endif
