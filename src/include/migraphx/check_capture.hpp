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
#ifndef MIGRAPHX_GUARD_CHECK_CAPTURE_HPP
#define MIGRAPHX_GUARD_CHECK_CAPTURE_HPP

#include <iterator>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace check_capture_detail {

// Operator-decomposing expression capture, modeled after `test/include/test.hpp`.
// `MIGRAPHX_CHECK_CAPTURE(expr)` returns an object whose `value()` evaluates the
// original predicate and whose `operator<<` prints the operator-decomposed form
// with the *actual* operand values, so a failed `MIGRAPHX_EXPECT` can report
// both the symbolic check and the runtime values that violated it.

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

// clang-format off
// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_FOREACH_BINARY_OPERATORS(m) \
    m(==, equal) \
    m(!=, not_equal) \
    m(<=, less_than_equal) \
    m(>=, greater_than_equal) \
    m(<, less_than) \
    m(>, greater_than) \
    m(and, and_op) \
    m(or, or_op)
// clang-format on

// clang-format off
// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_FOREACH_UNARY_OPERATORS(m) \
    m(not, not_op)
// clang-format on

// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_EACH_BINARY_OPERATOR_OBJECT(op, name) \
    struct name                                              \
    {                                                        \
        static std::string as_string() { return #op; }       \
        template <class T, class U>                          \
        static decltype(auto) call(T&& x, U&& y)             \
        {                                                    \
            return x op y;                                   \
        }                                                    \
    };

// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_EACH_UNARY_OPERATOR_OBJECT(op, name) \
    struct name                                             \
    {                                                       \
        static std::string as_string() { return #op; }      \
        template <class T>                                  \
        static decltype(auto) call(T&& x)                   \
        {                                                   \
            return op x;                                    \
        }                                                   \
    };

MIGRAPHX_CHECK_FOREACH_BINARY_OPERATORS(MIGRAPHX_CHECK_EACH_BINARY_OPERATOR_OBJECT)
MIGRAPHX_CHECK_FOREACH_UNARY_OPERATORS(MIGRAPHX_CHECK_EACH_UNARY_OPERATOR_OBJECT)

struct nop
{
    static std::string as_string() { return ""; }
    template <class T>
    static T call(T&& x)
    {
        return static_cast<T&&>(x);
    }
};

template <class Stream, class T>
void print_stream(Stream& s, const T& x);

template <class Stream, class T>
Stream& print_stream_impl(rank<0>, Stream& s, const T&)
{
    s << '?';
    return s;
}

template <class Stream, class Range>
auto print_stream_impl(rank<1>, Stream& s, const Range& v)
    -> decltype(v.end(), print_stream(s, *v.begin()), void())
{
    auto start = v.begin();
    auto last  = v.end();
    s << "{ ";
    if(start != last)
    {
        print_stream(s, *start);
        for(auto it = std::next(start); it != last; ++it)
        {
            s << ", ";
            print_stream(s, *it);
        }
    }
    s << "}";
}

template <class Stream, class T>
auto print_stream_impl(rank<2>, Stream& s, const T& x)
    -> decltype(s << x) // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
{
    if constexpr(std::is_pointer<T>{})
    {
        return s << static_cast<const void*>(x);
    }
    else if constexpr(std::is_same<T, bool>{})
    {
        s << (x ? "true" : "false");
        return s;
    }
    else
    {
        return s << x;
    }
}

template <class Stream, class T, class U>
Stream& print_stream_impl(rank<3>, Stream& s, const std::pair<T, U>& p)
{
    s << "{";
    print_stream(s, p.first);
    s << ", ";
    print_stream(s, p.second);
    s << "}";
    return s;
}

template <class Stream>
Stream& print_stream_impl(rank<4>, Stream& s, std::nullptr_t)
{
    s << "nullptr";
    return s;
}

template <class Stream, class T>
void print_stream(Stream& s, const T& x)
{
    print_stream_impl(rank<5>{}, s, x);
}

template <class T>
const T& get_value(const T& x)
{
    return x;
}

template <class T, class Operator = nop>
struct lhs_expression;

template <class T>
lhs_expression<T> make_lhs_expression(T&& lhs);

template <class T, class Operator>
lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator);

// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_EXPR_BINARY_OPERATOR(op, name)                                        \
    template <class V>                                                                       \
    friend auto operator op(self_t lhs2, V&& rhs2) /* NOLINT */                              \
    {                                                                                        \
        return make_expression(std::move(lhs2), std::forward<V>(rhs2), name{}); /* NOLINT */ \
    }

// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_EXPR_UNARY_OPERATOR(op, name)                            \
    friend auto operator op(self_t self) /* NOLINT */                           \
    {                                                                           \
        return make_lhs_expression(static_cast<decltype(self.lhs)&&>(self.lhs), \
                                   name{}); /* NOLINT */                        \
    }

template <class T, class U, class Operator>
struct expression
{
    using self_t = expression;
    T lhs;
    U rhs;

    friend std::ostream& operator<<(std::ostream& s, const expression& self)
    {
        print_stream(s, self.lhs);
        s << " " << Operator::as_string() << " ";
        print_stream(s, self.rhs);
        return s;
    }

    friend decltype(auto) get_value(const expression& e) { return e.value(); }

    decltype(auto) value() const { return Operator::call(get_value(lhs), get_value(rhs)); }

    MIGRAPHX_CHECK_FOREACH_UNARY_OPERATORS(MIGRAPHX_CHECK_EXPR_UNARY_OPERATOR)
    MIGRAPHX_CHECK_FOREACH_BINARY_OPERATORS(MIGRAPHX_CHECK_EXPR_BINARY_OPERATOR)
};

template <class T, class U, class Operator>
expression<T, U, Operator> make_expression(T&& lhs, U&& rhs, Operator)
{
    return {std::forward<T>(lhs), std::forward<U>(rhs)};
}

template <class T>
lhs_expression<T> make_lhs_expression(T&& lhs)
{
    return lhs_expression<T>{std::forward<T>(lhs)};
}

template <class T, class Operator>
lhs_expression<T, Operator> make_lhs_expression(T&& lhs, Operator)
{
    return lhs_expression<T, Operator>{std::forward<T>(lhs)};
}

template <class T, class Operator>
struct lhs_expression
{
    using self_t = lhs_expression;
    T lhs;
    explicit lhs_expression(T e) : lhs(static_cast<T&&>(e)) {}

    friend std::ostream& operator<<(std::ostream& s, const lhs_expression& self)
    {
        std::string op = Operator::as_string();
        if(not op.empty())
            s << op << " ";
        print_stream(s, self.lhs);
        return s;
    }

    friend decltype(auto) get_value(const lhs_expression& e) { return e.value(); }

    decltype(auto) value() const { return Operator::call(get_value(lhs)); }

    MIGRAPHX_CHECK_FOREACH_BINARY_OPERATORS(MIGRAPHX_CHECK_EXPR_BINARY_OPERATOR)
    MIGRAPHX_CHECK_FOREACH_UNARY_OPERATORS(MIGRAPHX_CHECK_EXPR_UNARY_OPERATOR)

// Arithmetic/bitwise operators bind tighter than comparison/logical ones, so
// after `capture->*a` we may still need `a + b >= c`-style chains to evaluate
// the arithmetic eagerly and re-wrap the result as a fresh `lhs_expression`.
// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_LHS_REOPERATOR(op)       \
    template <class U>                          \
    auto operator op(const U& rhs) const        \
    {                                           \
        return make_lhs_expression(lhs op rhs); \
    }
    MIGRAPHX_CHECK_LHS_REOPERATOR(+)
    MIGRAPHX_CHECK_LHS_REOPERATOR(-)
    MIGRAPHX_CHECK_LHS_REOPERATOR(*)
    MIGRAPHX_CHECK_LHS_REOPERATOR(/)
    MIGRAPHX_CHECK_LHS_REOPERATOR(%)
    MIGRAPHX_CHECK_LHS_REOPERATOR(&)
    MIGRAPHX_CHECK_LHS_REOPERATOR(|)
    MIGRAPHX_CHECK_LHS_REOPERATOR(^)
};

struct capture
{
    template <class T>
    auto operator->*(T&& x) const
    {
        return make_lhs_expression(std::forward<T>(x));
    }
};

template <class Expr>
std::string expression_to_string(const Expr& e)
{
    std::ostringstream ss;
    ss << e;
    return ss.str();
}

} // namespace check_capture_detail
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

// Operator precedence of `->*` is higher than the comparison/logical operators
// captured above, so `MIGRAPHX_CHECK_CAPTURE(a == b)` first wraps `a` in
// `lhs_expression`, then the overloaded `==` produces an `expression{a, b, ==}`.
// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_CAPTURE(...) ::migraphx::check_capture_detail::capture{}->*__VA_ARGS__

#endif
