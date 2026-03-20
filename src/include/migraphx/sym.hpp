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

inline value value_min(const value& a, const value& b)
{
    return value_invoke_common([](auto x, auto y) { return x < y ? x : y; }, a, b);
}

inline value value_max(const value& a, const value& b)
{
    return value_invoke_common([](auto x, auto y) { return x > y ? x : y; }, a, b);
}

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

    friend interval operator+(interval a, interval b)
    {
        auto f = [](auto x, auto y) { return x + y; };
        return {value_invoke_common(f, a.min, b.min), value_invoke_common(f, a.max, b.max)};
    }

    friend interval operator-(interval a, interval b)
    {
        auto f = [](auto x, auto y) { return x - y; };
        return {value_invoke_common(f, a.min, b.max), value_invoke_common(f, a.max, b.min)};
    }

    friend interval operator*(interval a, interval b)
    {
        auto f  = [](auto x, auto y) { return x * y; };
        auto p1 = value_invoke_common(f, a.min, b.min);
        auto p2 = value_invoke_common(f, a.min, b.max);
        auto p3 = value_invoke_common(f, a.max, b.min);
        auto p4 = value_invoke_common(f, a.max, b.max);
        return {value_min(value_min(p1, p2), value_min(p3, p4)),
                value_max(value_max(p1, p2), value_max(p3, p4))};
    }

    friend interval operator/(interval a, interval b)
    {
        auto f  = [](auto x, auto y) { return x / y; };
        auto p1 = value_invoke_common(f, a.min, b.min);
        auto p2 = value_invoke_common(f, a.min, b.max);
        auto p3 = value_invoke_common(f, a.max, b.min);
        auto p4 = value_invoke_common(f, a.max, b.max);
        return {value_min(value_min(p1, p2), value_min(p3, p4)),
                value_max(value_max(p1, p2), value_max(p3, p4))};
    }

    friend interval operator-(interval a)
    {
        auto f = [](auto x) { return -x; };
        return {value_invoke_common(f, a.max), value_invoke_common(f, a.min)};
    }

    friend bool operator==(const interval& a, const interval& b)
    {
        return a.min == b.min and a.max == b.max;
    }

    friend bool operator!=(const interval& a, const interval& b) { return not(a == b); }
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
};

struct variable_node
{
    std::string name;
    std::vector<interval> constraints;
};

struct op_node
{
    const op_def* op;
};

using node_variant = std::variant<literal_node, variable_node, op_node>;

class MIGRAPHX_EXPORT expr
{
    struct impl;
    std::shared_ptr<const impl> pimpl;

    public:
    expr() = default;

    template <class Node>
    explicit expr(Node node, std::vector<expr> children = {});

    value eval(const std::unordered_map<std::string, value>& vars) const;
    interval eval_interval(const std::unordered_map<std::string, interval>& vars) const;
};

struct expr::impl
{
    node_variant node;
    std::vector<expr> children;
};

template <class Node>
expr::expr(Node node, std::vector<expr> children)
    : pimpl(std::make_shared<impl>(impl{node_variant{std::move(node)}, std::move(children)}))
{
}

template <class T>
auto lit(T v) -> std::enable_if_t<std::is_arithmetic_v<T>, expr>
{
    if constexpr(std::is_integral_v<T>)
        return expr(literal_node{value{static_cast<int64_t>(v)}});
    else
        return expr(literal_node{value{static_cast<double>(v)}});
}

inline expr var(std::string name) { return expr(variable_node{std::move(name), {}}); }

inline expr var(std::string name, interval constraint)
{
    return expr(variable_node{std::move(name), {std::move(constraint)}});
}

inline expr arg(expr x) { return x; }

template <class T>
auto arg(T x) -> std::enable_if_t<std::is_arithmetic_v<T>, expr>
{
    if constexpr(std::is_integral_v<T>)
        return lit(static_cast<int64_t>(x));
    else
        return lit(static_cast<double>(x));
}

inline expr call_op(const op_def* op, std::vector<expr> args)
{
    return expr(op_node{op}, std::move(args));
}

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

inline expr operator+(expr ex, expr ey)
{
    return call("+", [](auto x, auto y) { return x + y; })(std::move(ex), std::move(ey));
}

inline expr operator-(expr ex, expr ey)
{
    return call("-", [](auto x, auto y) { return x - y; })(std::move(ex), std::move(ey));
}

inline expr operator*(expr ex, expr ey)
{
    return call("*", [](auto x, auto y) { return x * y; })(std::move(ex), std::move(ey));
}

inline expr operator/(expr ex, expr ey)
{
    return call("/", [](auto x, auto y) { return x / y; })(std::move(ex), std::move(ey));
}

inline expr operator-(expr e)
{
    return call("neg", [](auto x) { return -x; })(std::move(e));
}

inline expr sqrt(expr e)
{
    return call(
        "sqrt",
        MIGRAPHX_LIFT(std::sqrt),
        [](interval x) -> interval {
            auto lo = std::sqrt(std::max(0.0, to<double>(x.min)));
            auto hi = std::sqrt(std::max(0.0, to<double>(x.max)));
            return {lo, hi};
        })(std::move(e));
}

} // namespace sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
