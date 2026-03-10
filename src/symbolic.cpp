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

#include <migraphx/symbolic.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>

#include <symengine/visitor.h>

#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

symbolic_dim::symbolic_dim(std::size_t value)
    : expr(SymEngine::integer(value)), min(value), max(value)
{
}

symbolic_dim::symbolic_dim(const std::string& name,
                           std::size_t min_val,
                           std::size_t max_val,
                           std::set<std::size_t> opt)
    : expr(SymEngine::symbol(name)), min(min_val), max(max_val), optimals(std::move(opt))
{
}

symbolic_dim::symbolic_dim(SymEngine::Expression e,
                           std::size_t min_val,
                           std::size_t max_val,
                           std::set<std::size_t> opt)
    : expr(std::move(e)), min(min_val), max(max_val), optimals(std::move(opt))
{
}

symbolic_dim::symbolic_dim(const shape::dynamic_dimension& dd)
    : expr(SymEngine::integer(dd.is_fixed() ? dd.min : 0)),
      min(dd.min),
      max(dd.max),
      optimals(dd.optimals)
{
}

bool symbolic_dim::is_fixed() const { return min == max; }

std::string symbolic_dim::to_string() const
{
    std::stringstream ss;
    ss << expr;
    return ss.str();
}

shape::dynamic_dimension symbolic_dim::to_dynamic_dimension() const
{
    return shape::dynamic_dimension{min, max, optimals};
}

std::optional<symbolic_dim> symbolic_dim::intersection(const symbolic_dim& other) const
{
    auto left  = std::max(this->min, other.min);
    auto right = std::min(this->max, other.max);
    if(left <= right)
    {
        auto result_expr = (not this->is_fixed()) ? this->expr : other.expr;
        return symbolic_dim{result_expr, left, right};
    }
    return std::nullopt;
}

// symbolic_dim + symbolic_dim
symbolic_dim& symbolic_dim::operator+=(const symbolic_dim& x)
{
    expr = expr + x.expr;
    min  = min + x.min;
    max  = (max > std::numeric_limits<std::size_t>::max() - x.max)
               ? std::numeric_limits<std::size_t>::max()
               : max + x.max;
    optimals.clear();
    return *this;
}

symbolic_dim& symbolic_dim::operator-=(const symbolic_dim& x)
{
    expr = expr - x.expr;
    min  = (min > x.max) ? min - x.max : 0;
    max  = (max > x.min) ? max - x.min : 0;
    optimals.clear();
    return *this;
}

symbolic_dim& symbolic_dim::operator*=(const symbolic_dim& x)
{
    expr = expr * x.expr;
    min  = min * x.min;
    max  = (max > std::numeric_limits<std::size_t>::max() / (x.max == 0 ? 1 : x.max))
               ? std::numeric_limits<std::size_t>::max()
               : max * x.max;
    optimals.clear();
    return *this;
}

symbolic_dim operator+(const symbolic_dim& x, const symbolic_dim& y)
{
    auto result = x;
    result += y;
    return result;
}

symbolic_dim operator-(const symbolic_dim& x, const symbolic_dim& y)
{
    auto result = x;
    result -= y;
    return result;
}

symbolic_dim operator*(const symbolic_dim& x, const symbolic_dim& y)
{
    auto result = x;
    result *= y;
    return result;
}

// symbolic_dim + size_t
symbolic_dim& symbolic_dim::operator+=(const std::size_t& x)
{
    expr = expr + SymEngine::Expression(SymEngine::integer(x));
    min += x;
    max = (max > std::numeric_limits<std::size_t>::max() - x)
              ? std::numeric_limits<std::size_t>::max()
              : max + x;
    optimals.clear();
    return *this;
}

symbolic_dim& symbolic_dim::operator-=(const std::size_t& x)
{
    expr = expr - SymEngine::Expression(SymEngine::integer(x));
    min  = (min > x) ? min - x : 0;
    max  = (max > x) ? max - x : 0;
    optimals.clear();
    return *this;
}

symbolic_dim& symbolic_dim::operator*=(const std::size_t& x)
{
    expr = expr * SymEngine::Expression(SymEngine::integer(x));
    min *= x;
    max = (x != 0 and max > std::numeric_limits<std::size_t>::max() / x)
              ? std::numeric_limits<std::size_t>::max()
              : max * x;
    optimals.clear();
    return *this;
}

symbolic_dim operator+(const symbolic_dim& x, const std::size_t& y)
{
    auto result = x;
    result += y;
    return result;
}

symbolic_dim operator+(const std::size_t& x, const symbolic_dim& y)
{
    return y + x;
}

symbolic_dim operator-(const symbolic_dim& x, const std::size_t& y)
{
    auto result = x;
    result -= y;
    return result;
}

symbolic_dim operator*(const symbolic_dim& x, const std::size_t& y)
{
    auto result = x;
    result *= y;
    return result;
}

symbolic_dim operator*(const std::size_t& x, const symbolic_dim& y)
{
    return y * x;
}

bool operator==(const symbolic_dim& x, const symbolic_dim& y)
{
    return x.expr == y.expr and x.min == y.min and x.max == y.max;
}

bool operator!=(const symbolic_dim& x, const symbolic_dim& y) { return not(x == y); }

std::ostream& operator<<(std::ostream& os, const symbolic_dim& x)
{
    os << x.to_string();
    if(not x.is_fixed())
        os << "[" << x.min << ".." << x.max << "]";
    return os;
}

void migraphx_to_value(value& v, const symbolic_dim& sd)
{
    value result   = value::object{};
    result["expr"] = migraphx::to_value(sd.to_string());
    result["min"]  = migraphx::to_value(sd.min);
    result["max"]  = migraphx::to_value(sd.max);
    result["optimals"] = migraphx::to_value(sd.optimals);
    v = result;
}

void migraphx_from_value(const value& v, symbolic_dim& sd)
{
    auto expr_str = v.at("expr").get_string();
    sd.expr       = SymEngine::Expression(expr_str);
    sd.min        = from_value<std::size_t>(v.at("min"));
    sd.max        = from_value<std::size_t>(v.at("max"));
    sd.optimals   = from_value<std::set<std::size_t>>(v.at("optimals"));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
