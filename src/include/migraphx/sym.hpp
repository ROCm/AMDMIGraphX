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
#ifndef MIGRAPHX_GUARD_SYM_HPP
#define MIGRAPHX_GUARD_SYM_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include <migraphx/config.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/requires.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace sym {

using value = std::variant<int64_t, double>;

template <class To>
To to(const value& v)
{
    return std::visit([](auto x) -> To { return x; }, v);
}

template <class F, class... Ts>
value value_invoke(F f, const Ts&... vs)
{
    return std::visit([&](auto... xs) -> value { return f(xs...); }, vs...);
}

template <class F, class... Ts>
value value_invoke_common(F f, const Ts&... xs)
{
    if((std::holds_alternative<int64_t>(xs) and ...))
        return f(std::get<int64_t>(xs)...);
    return f(to<double>(xs)...);
}

value value_min(const value& a, const value& b);
value value_max(const value& a, const value& b);

template <std::size_t N, class F>
auto unpack_container(F f)
{
    return [=](auto&& c) {
        if(c.size() != N)
            MIGRAPHX_THROW("Mismatch number of inputs");
        return sequence_c<N>([&](auto... is) { return f(c[is]...); });
    };
}

struct interval
{
    value min = int64_t{0};
    value max = int64_t{0};

    interval& operator+=(interval b) { return *this = *this + b; }
    interval& operator-=(interval b) { return *this = *this - b; }
    interval& operator*=(interval b) { return *this = *this * b; }
    interval& operator/=(interval b) { return *this = *this / b; }

    friend interval operator+(interval a, interval b);
    friend interval operator-(interval a, interval b);
    friend interval operator*(interval a, interval b);
    friend interval operator/(interval a, interval b);
    friend interval operator-(interval a);
    friend bool operator<(interval a, interval b);
    friend bool operator<=(interval a, interval b);
    friend bool operator>(interval a, interval b);
    friend bool operator>=(interval a, interval b);
    friend bool operator==(const interval& a, const interval& b);
    friend bool operator!=(const interval& a, const interval& b);
    friend interval sin(interval x);
    friend interval cos(interval x);
    friend interval tan(interval x);
    friend interval exp(interval x);
    friend interval log(interval x);
    friend interval sqrt(interval x);
    friend interval abs(interval x);
    friend interval floor(interval x);
    friend interval ceil(interval x);
    friend interval pow(interval x, interval y);
    friend interval min(interval x, interval y);
    friend interval max(interval x, interval y);
};

struct op_def
{
    std::string name;
    std::function<value(const std::vector<value>&)> eval;
    std::function<interval(const std::vector<interval>&)> eval_interval;
};

struct literal_node
{
    value val;
    friend bool operator==(const literal_node& a, const literal_node& b) { return a.val == b.val; }
    friend bool operator!=(const literal_node& a, const literal_node& b) { return not(a == b); }
};

struct variable_node
{
    std::string name;
    std::vector<interval> constraints;
    friend bool operator==(const variable_node& a, const variable_node& b)
    {
        return a.name == b.name and a.constraints == b.constraints;
    }
    friend bool operator!=(const variable_node& a, const variable_node& b) { return not(a == b); }
};

struct op_node
{
    const op_def* op;
    friend bool operator==(const op_node& a, const op_node& b) { return a.op == b.op; }
    friend bool operator!=(const op_node& a, const op_node& b) { return not(a == b); }
};

using node_variant = std::variant<literal_node, variable_node, op_node>;

class expr;
expr lit(value v);

class MIGRAPHX_EXPORT expr
{
    struct impl;
    std::shared_ptr<const impl> pimpl;
    static std::shared_ptr<const impl> make_impl(node_variant node, std::vector<expr> children);

    public:
    expr() = default;

    template <class Node>
    explicit expr(Node node, std::vector<expr> children = {})
        : pimpl(make_impl(node_variant{std::move(node)}, std::move(children)))
    {
    }

    std::string name() const;
    bool is_raw() const;
    const node_variant& node() const;
    const std::vector<expr>& children() const;
    value eval(const std::unordered_map<std::string, value>& vars) const;
    interval eval_interval(const std::unordered_map<std::string, interval>& vars) const;
    std::string to_string() const;

    friend expr operator+(expr ex, expr ey);
    friend expr operator-(expr ex, expr ey);
    friend expr operator*(expr ex, expr ey);
    friend expr operator/(expr ex, expr ey);
    friend expr operator-(expr e);
    friend bool operator==(const expr& a, const expr& b);
    friend bool operator!=(const expr& a, const expr& b);

#define MIGRAPHX_SYM_DEFINE_OP_TYPE(binary, assign, type) \
    expr& operator assign(type x) { return *this = *this binary lit(x); } \
    friend expr operator binary(expr ex, type y) { return ex binary lit(y); } \
    friend expr operator binary(type x, expr ey) { return lit(x) binary ey; }
    
#define MIGRAPHX_SYM_DEFINE_OP(binary, assign) \
    expr& operator assign(expr ey) { return *this = *this binary std::move(ey); } \
    MIGRAPHX_SYM_DEFINE_OP_TYPE(binary, assign, int64_t) \
    MIGRAPHX_SYM_DEFINE_OP_TYPE(binary, assign, double)

    MIGRAPHX_SYM_DEFINE_OP(+, +=)
    MIGRAPHX_SYM_DEFINE_OP(-, -=)
    MIGRAPHX_SYM_DEFINE_OP(*, *=)
    MIGRAPHX_SYM_DEFINE_OP(/, /=)
};

expr lit(value v);

template <class T, MIGRAPHX_REQUIRES(std::is_arithmetic<T>{})>
expr lit(T v)
{
    if constexpr(std::is_integral<T>{})
        return lit(value{int64_t{v}});
    else
        return lit(value{double{v}});
}

expr var(std::string name);
expr var(std::string name, interval constraint);

expr arg(expr x);

template <class T, MIGRAPHX_REQUIRES(std::is_arithmetic<T>{})>
expr arg(T x)
{
    if constexpr(std::is_integral<T>{})
        return lit(static_cast<int64_t>(x));
    else
        return lit(static_cast<double>(x));
}

expr call_op(const op_def* op, std::vector<expr> args);

template <class Eval, class EvalInterval>
expr call_op(std::string name, Eval eval, EvalInterval eval_interval, std::vector<expr> args)
{
    static const op_def op{std::move(name), std::move(eval), std::move(eval_interval)};
    return call_op(&op, std::move(args));
}

template <class Eval, class EvalInterval>
auto call(std::string name, Eval eval, EvalInterval eval_interval)
{
    return [=](auto... es) {
        auto eval1 = unpack_container<sizeof...(es)>(
            [=](auto... xs) { return value_invoke_common(eval, xs...); });
        auto eval_interval1 =
            unpack_container<sizeof...(es)>([=](auto... xs) { return eval_interval(xs...); });
        return call_op(name, eval1, eval_interval1, {arg(es)...});
    };
}

template <class Eval>
auto call(std::string name, Eval eval)
{
    return call(name, eval, eval);
}

std::string to_string(const expr& e);

expr parse(const std::string& str);

expr sin(expr e);
expr cos(expr e);
expr tan(expr e);
expr exp(expr e);
expr log(expr e);
expr sqrt(expr e);
expr abs(expr e);
expr floor(expr e);
expr ceil(expr e);
expr pow(expr x, expr y);
expr min(expr x, expr y);
expr max(expr x, expr y);

// Pattern matching rewrite DSL
expr pvar(int id);

struct rewrite_rule
{
    expr pattern;
    expr replacement;
};

inline rewrite_rule operator>>(expr pattern, expr replacement)
{
    return {std::move(pattern), std::move(replacement)};
}

template <class T, MIGRAPHX_REQUIRES(std::is_arithmetic<T>{})>
rewrite_rule operator>>(expr pattern, T replacement)
{
    return {std::move(pattern), lit(replacement)};
}

expr simplify(expr e, std::vector<rewrite_rule> rules);

struct simplifier
{
    expr e;
    template <class... Rules>
    expr operator()(Rules... rules) const
    {
        return sym::simplify(e, {std::move(rules)...});
    }
};

inline simplifier simplify(expr e) { return {std::move(e)}; }

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
