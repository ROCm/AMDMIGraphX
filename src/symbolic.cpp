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
#include <migraphx/shape.hpp>
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>

#include <symengine/expression.h>
#include <symengine/integer.h>
#include <symengine/symbol.h>
#include <symengine/visitor.h>
#include <symengine/parser.h>

#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// ---------------------------------------------------------------------------
// symbolic_expr pimpl
// ---------------------------------------------------------------------------

struct symbolic_expr::impl
{
    SymEngine::Expression expr;

    impl() : expr(0) {}
    explicit impl(SymEngine::Expression e) : expr(std::move(e)) {}
};

symbolic_expr::symbolic_expr() = default;

symbolic_expr::symbolic_expr(std::shared_ptr<const impl> pi) : p(std::move(pi)) {}

symbolic_expr::symbolic_expr(std::size_t n)
    : p(std::make_shared<impl>(SymEngine::Expression(SymEngine::integer(n))))
{
}

symbolic_expr::symbolic_expr(const std::string& s)
{
    if(s.empty())
        return;
    p = std::make_shared<impl>(SymEngine::Expression(SymEngine::parse(s)));
}

bool symbolic_expr::empty() const { return p == nullptr; }

std::string symbolic_expr::to_string() const
{
    if(empty())
        return {};
    std::stringstream ss;
    ss << p->expr;
    return ss.str();
}

symbolic_expr operator+(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->expr : SymEngine::Expression(0);
    auto eb = b.p ? b.p->expr : SymEngine::Expression(0);
    return {std::make_shared<symbolic_expr::impl>(ea + eb)};
}

symbolic_expr operator-(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->expr : SymEngine::Expression(0);
    auto eb = b.p ? b.p->expr : SymEngine::Expression(0);
    return {std::make_shared<symbolic_expr::impl>(ea - eb)};
}

symbolic_expr operator*(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->expr : SymEngine::Expression(0);
    auto eb = b.p ? b.p->expr : SymEngine::Expression(0);
    return {std::make_shared<symbolic_expr::impl>(ea * eb)};
}

symbolic_expr operator/(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return {};
    auto ea = a.p ? a.p->expr : SymEngine::Expression(0);
    auto eb = b.p ? b.p->expr : SymEngine::Expression(0);
    return {std::make_shared<symbolic_expr::impl>(ea / eb)};
}

bool operator==(const symbolic_expr& a, const symbolic_expr& b)
{
    if(a.empty() and b.empty())
        return true;
    if(a.empty() != b.empty())
        return false;
    return SymEngine::eq(*a.p->expr.get_basic(), *b.p->expr.get_basic());
}

bool operator!=(const symbolic_expr& a, const symbolic_expr& b) { return not(a == b); }

std::ostream& operator<<(std::ostream& os, const symbolic_expr& e)
{
    if(not e.empty())
        os << e.p->expr;
    return os;
}

void migraphx_to_value(value& v, const symbolic_expr& e)
{
    v = migraphx::to_value(e.to_string());
}

void migraphx_from_value(const value& v, symbolic_expr& e)
{
    auto s = v.get_string();
    if(s.empty())
        e = symbolic_expr{};
    else
        e = symbolic_expr(s);
}

// ---------------------------------------------------------------------------
// dynamic_dimension dd-to-dd arithmetic
// ---------------------------------------------------------------------------

using dd = shape::dynamic_dimension;

dd& dd::operator+=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) + x.sym.value_or(symbolic_expr(x.min));
    min    = min + x.min;
    max    = (max > std::numeric_limits<std::size_t>::max() - x.max)
                 ? std::numeric_limits<std::size_t>::max()
                 : max + x.max;
    optimals.clear();
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator-=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) - x.sym.value_or(symbolic_expr(x.min));
    min    = (min > x.max) ? min - x.max : 0;
    max    = (max > x.min) ? max - x.min : 0;
    optimals.clear();
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator*=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) * x.sym.value_or(symbolic_expr(x.min));
    min    = min * x.min;
    max    = (max > std::numeric_limits<std::size_t>::max() / (x.max == 0 ? 1 : x.max))
                 ? std::numeric_limits<std::size_t>::max()
                 : max * x.max;
    optimals.clear();
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd& dd::operator/=(const dd& x)
{
    auto e = sym.value_or(symbolic_expr(min)) / x.sym.value_or(symbolic_expr(x.min));
    min    = (x.max == 0) ? 0 : min / x.max;
    max    = (x.min == 0) ? std::numeric_limits<std::size_t>::max() : max / x.min;
    optimals.clear();
    sym = (sym or x.sym) ? optional<symbolic_expr>(e) : nullopt;
    return *this;
}

dd operator+(const dd& x, const dd& y)
{
    auto result = x;
    result += y;
    return result;
}

dd operator-(const dd& x, const dd& y)
{
    auto result = x;
    result -= y;
    return result;
}

dd operator*(const dd& x, const dd& y)
{
    auto result = x;
    result *= y;
    return result;
}

dd operator/(const dd& x, const dd& y)
{
    auto result = x;
    result /= y;
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
