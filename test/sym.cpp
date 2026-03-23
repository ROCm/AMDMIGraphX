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
#include <migraphx/sym.hpp>
#include <test.hpp>

using migraphx::sym::abs;
using migraphx::sym::arg;
using migraphx::sym::call;
using migraphx::sym::ceil;
using migraphx::sym::cos;
using migraphx::sym::exp;
using migraphx::sym::expr;
using migraphx::sym::floor;
using migraphx::sym::interval;
using migraphx::sym::lit;
using migraphx::sym::log;
using migraphx::sym::max;
using migraphx::sym::min;
using migraphx::sym::pow;
using migraphx::sym::sin;
using migraphx::sym::sqrt;
using migraphx::sym::tan;
using migraphx::sym::to_string;
using migraphx::sym::value;
using migraphx::sym::var;

// ---- Value evaluation tests ----

TEST_CASE(literal_int_eval)
{
    auto e      = lit(42);
    auto result = e.eval({});
    EXPECT(result == value{int64_t{42}});
}

TEST_CASE(literal_double_eval)
{
    auto e      = lit(3.14);
    auto result = e.eval({});
    EXPECT(result == value{3.14});
}

TEST_CASE(variable_eval)
{
    auto e      = var("x");
    auto result = e.eval({{"x", int64_t{10}}});
    EXPECT(result == value{int64_t{10}});
}

TEST_CASE(add_int_eval)
{
    auto e      = lit(3) + lit(4);
    auto result = e.eval({});
    EXPECT(result == value{int64_t{7}});
}

TEST_CASE(add_mixed_eval)
{
    auto e      = lit(3) + lit(1.5);
    auto result = e.eval({});
    EXPECT(result == value{4.5});
}

TEST_CASE(sub_eval)
{
    auto e      = lit(10) - lit(3);
    auto result = e.eval({});
    EXPECT(result == value{int64_t{7}});
}

TEST_CASE(mul_eval)
{
    auto e      = lit(6) * lit(7);
    auto result = e.eval({});
    EXPECT(result == value{int64_t{42}});
}

TEST_CASE(div_double_eval)
{
    auto e      = lit(10.0) / lit(4.0);
    auto result = e.eval({});
    EXPECT(result == value{2.5});
}

TEST_CASE(neg_eval)
{
    auto e      = -lit(5);
    auto result = e.eval({});
    EXPECT(result == value{int64_t{-5}});
}

TEST_CASE(compound_expr_eval)
{
    auto x      = var("x");
    auto e      = (x + lit(3)) * lit(2);
    auto result = e.eval({{"x", int64_t{5}}});
    EXPECT(result == value{int64_t{16}});
}

TEST_CASE(multi_variable_eval)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x * y + lit(1);

    auto result = e.eval({{"x", int64_t{3}}, {"y", int64_t{4}}});
    EXPECT(result == value{int64_t{13}});
}

TEST_CASE(sqrt_eval)
{
    auto e      = sqrt(lit(4.0));
    auto result = e.eval({});
    EXPECT(result == value{2.0});
}

TEST_CASE(sqrt_int_eval)
{
    auto e      = sqrt(lit(9));
    auto result = e.eval({});
    EXPECT(result == value{3.0});
}

TEST_CASE(nested_sqrt_eval)
{
    auto e      = sqrt(lit(16.0)) + lit(1.0);
    auto result = e.eval({});
    EXPECT(result == value{5.0});
}

TEST_CASE(arg_int_literal)
{
    auto x      = var("x");
    auto e      = call("+", [](auto a, auto b) { return a + b; })(x, 3);
    auto result = e.eval({{"x", int64_t{5}}});
    EXPECT(result == value{int64_t{8}});
}

TEST_CASE(arg_double_literal)
{
    auto x      = var("x");
    auto e      = call("*", [](auto a, auto b) { return a * b; })(x, 2.0);
    auto result = e.eval({{"x", 3.0}});
    EXPECT(result == value{6.0});
}

TEST_CASE(shared_subexpr)
{
    auto x      = var("x");
    auto sub    = x + lit(1);
    auto e      = sub * sub;
    auto result = e.eval({{"x", int64_t{4}}});
    EXPECT(result == value{int64_t{25}});
}

// ---- Interval evaluation tests ----

TEST_CASE(literal_interval)
{
    auto e      = lit(5);
    auto result = e.eval_interval({});
    EXPECT(result == interval{int64_t{5}, int64_t{5}});
}

TEST_CASE(literal_double_interval)
{
    auto e      = lit(2.5);
    auto result = e.eval_interval({});
    EXPECT(result == interval{2.5, 2.5});
}

TEST_CASE(variable_interval)
{
    auto x      = var("x");
    auto result = x.eval_interval({{"x", interval{int64_t{1}, int64_t{10}}}});
    EXPECT(result == (interval{int64_t{1}, int64_t{10}}));
}

TEST_CASE(add_interval)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x + y;
    // [1,3] + [2,4] = [3,7]
    auto result = e.eval_interval(
        {{"x", interval{int64_t{1}, int64_t{3}}}, {"y", interval{int64_t{2}, int64_t{4}}}});
    EXPECT(result == (interval{int64_t{3}, int64_t{7}}));
}

TEST_CASE(sub_interval)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x - y;
    // [5,10] - [1,3] = [5-3, 10-1] = [2, 9]
    auto result = e.eval_interval(
        {{"x", interval{int64_t{5}, int64_t{10}}}, {"y", interval{int64_t{1}, int64_t{3}}}});
    EXPECT(result == (interval{int64_t{2}, int64_t{9}}));
}

TEST_CASE(mul_interval_positive)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x * y;
    // [2,3] * [4,5]: products = {8,10,12,15}, min=8, max=15
    auto result = e.eval_interval(
        {{"x", interval{int64_t{2}, int64_t{3}}}, {"y", interval{int64_t{4}, int64_t{5}}}});
    EXPECT(result == (interval{int64_t{8}, int64_t{15}}));
}

TEST_CASE(mul_interval_mixed_sign)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x * y;
    // [-2,3] * [1,4]: products = {-2,-8,3,12}, min=-8, max=12
    auto result = e.eval_interval(
        {{"x", interval{int64_t{-2}, int64_t{3}}}, {"y", interval{int64_t{1}, int64_t{4}}}});
    EXPECT(result == (interval{int64_t{-8}, int64_t{12}}));
}

TEST_CASE(neg_interval)
{
    auto x = var("x");
    auto e = -x;
    // -[3,7] = [-7,-3]
    auto result = e.eval_interval({{"x", interval{int64_t{3}, int64_t{7}}}});
    EXPECT(result == (interval{int64_t{-7}, int64_t{-3}}));
}

TEST_CASE(compound_interval)
{
    auto x = var("x");
    auto e = (x + lit(3)) * lit(2);
    // x in [1,5], x+3 in [4,8], *2 in [8,16]
    auto result = e.eval_interval({{"x", interval{int64_t{1}, int64_t{5}}}});
    EXPECT(result == (interval{int64_t{8}, int64_t{16}}));
}

TEST_CASE(sqrt_interval)
{
    auto e      = sqrt(var("x"));
    auto result = e.eval_interval({{"x", interval{4.0, 9.0}}});
    EXPECT(result == (interval{2.0, 3.0}));
}

TEST_CASE(variable_constraint_interval)
{
    auto x      = var("x", interval{int64_t{0}, int64_t{100}});
    auto result = x.eval_interval({});
    EXPECT(result == (interval{int64_t{0}, int64_t{100}}));
}

TEST_CASE(constraint_overridden_by_map)
{
    auto x      = var("x", interval{int64_t{0}, int64_t{100}});
    auto result = x.eval_interval({{"x", interval{int64_t{5}, int64_t{10}}}});
    EXPECT(result == (interval{int64_t{5}, int64_t{10}}));
}

// ---- Interval comparison tests ----

TEST_CASE(interval_less_true)
{
    // [1,3] < [5,7] → true (a.max < b.min)
    interval a{int64_t{1}, int64_t{3}};
    interval b{int64_t{5}, int64_t{7}};
    EXPECT(a < b);
}

TEST_CASE(interval_less_false)
{
    // [5,7] < [1,3] → false
    interval a{int64_t{5}, int64_t{7}};
    interval b{int64_t{1}, int64_t{3}};
    EXPECT(not(a < b));
}

TEST_CASE(interval_less_overlapping)
{
    // [1,5] < [3,7] → false (not all values satisfy)
    interval a{int64_t{1}, int64_t{5}};
    interval b{int64_t{3}, int64_t{7}};
    EXPECT(not(a < b));
}

TEST_CASE(interval_less_equal_endpoint)
{
    // [1,5] < [5,10] → false (a.max == b.min, not strictly less)
    interval a{int64_t{1}, int64_t{5}};
    interval b{int64_t{5}, int64_t{10}};
    EXPECT(not(a < b));
}

TEST_CASE(interval_leq_true)
{
    // [1,3] <= [3,5] → true (a.max <= b.min)
    interval a{int64_t{1}, int64_t{3}};
    interval b{int64_t{3}, int64_t{5}};
    EXPECT(a <= b);
}

TEST_CASE(interval_leq_false)
{
    // [5,7] <= [1,3] → false
    interval a{int64_t{5}, int64_t{7}};
    interval b{int64_t{1}, int64_t{3}};
    EXPECT(not(a <= b));
}

TEST_CASE(interval_leq_overlapping)
{
    // [1,5] <= [3,4] → false (not all values satisfy)
    interval a{int64_t{1}, int64_t{5}};
    interval b{int64_t{3}, int64_t{4}};
    EXPECT(not(a <= b));
}

TEST_CASE(interval_greater_true)
{
    // [5,7] > [1,3] → true (a.min > b.max)
    interval a{int64_t{5}, int64_t{7}};
    interval b{int64_t{1}, int64_t{3}};
    EXPECT(a > b);
}

TEST_CASE(interval_greater_false)
{
    // [1,3] > [5,7] → false
    interval a{int64_t{1}, int64_t{3}};
    interval b{int64_t{5}, int64_t{7}};
    EXPECT(not(a > b));
}

TEST_CASE(interval_geq_true)
{
    // [3,5] >= [1,3] → true (a.min >= b.max)
    interval a{int64_t{3}, int64_t{5}};
    interval b{int64_t{1}, int64_t{3}};
    EXPECT(a >= b);
}

TEST_CASE(interval_geq_false)
{
    // [1,3] >= [5,7] → false
    interval a{int64_t{1}, int64_t{3}};
    interval b{int64_t{5}, int64_t{7}};
    EXPECT(not(a >= b));
}

TEST_CASE(interval_geq_overlapping)
{
    // [1,5] >= [3,7] → false (not all values satisfy)
    interval a{int64_t{1}, int64_t{5}};
    interval b{int64_t{3}, int64_t{7}};
    EXPECT(not(a >= b));
}

// ---- Interval compound assignment tests ----

TEST_CASE(interval_plus_assign)
{
    // [1,3] += [2,4] = [3,7]
    interval a{int64_t{1}, int64_t{3}};
    a += interval{int64_t{2}, int64_t{4}};
    EXPECT(a == (interval{int64_t{3}, int64_t{7}}));
}

TEST_CASE(interval_minus_assign)
{
    // [5,10] -= [1,3] = [2,9]
    interval a{int64_t{5}, int64_t{10}};
    a -= interval{int64_t{1}, int64_t{3}};
    EXPECT(a == (interval{int64_t{2}, int64_t{9}}));
}

TEST_CASE(interval_times_assign)
{
    // [2,3] *= [4,5] = [8,15]
    interval a{int64_t{2}, int64_t{3}};
    a *= interval{int64_t{4}, int64_t{5}};
    EXPECT(a == (interval{int64_t{8}, int64_t{15}}));
}

TEST_CASE(interval_div_assign)
{
    // [10.0,20.0] /= [2.0,5.0] = [2.0,10.0]
    interval a{2.0, 10.0};
    a /= interval{1.0, 5.0};
    EXPECT(a == (interval{0.4, 10.0}));
}

TEST_CASE(interval_compound_assign_no_alias)
{
    interval a{int64_t{1}, int64_t{3}};
    interval b = a;
    a += interval{int64_t{10}, int64_t{10}};
    // b unchanged
    EXPECT(b == (interval{int64_t{1}, int64_t{3}}));
    EXPECT(a == (interval{int64_t{11}, int64_t{13}}));
}

// ---- Expr structural equality tests ----

TEST_CASE(expr_equal_literals)
{
    EXPECT(lit(42) == lit(42));
    EXPECT(lit(3.14) == lit(3.14));
}

TEST_CASE(expr_different_literals)
{
    EXPECT(lit(1) != lit(2));
    EXPECT(lit(1) != lit(1.0));
}

TEST_CASE(expr_equal_variables) { EXPECT(var("x") == var("x")); }

TEST_CASE(expr_different_variables) { EXPECT(var("x") != var("y")); }

TEST_CASE(expr_variable_constraint_equality)
{
    auto c = interval{int64_t{0}, int64_t{10}};
    EXPECT(var("x", c) == var("x", c));
    EXPECT(var("x") != var("x", c));
}

TEST_CASE(expr_equal_compound)
{
    auto x = var("x");
    EXPECT(x + lit(1) == x + lit(1));
    EXPECT(x * lit(2) == x * lit(2));
}

TEST_CASE(expr_different_compound)
{
    auto x = var("x");
    EXPECT(x + lit(1) != x + lit(2));
    EXPECT(x + lit(1) != x * lit(1));
}

TEST_CASE(expr_shared_subexpr_identity)
{
    auto x   = var("x");
    auto sub = x + lit(1);
    EXPECT(sub == sub);
}

TEST_CASE(expr_default_constructed_equal)
{
    expr a;
    expr b;
    EXPECT(a == b);
}

TEST_CASE(expr_default_not_equal_to_lit)
{
    expr a;
    EXPECT(a != lit(0));
}

// ---- Compound assignment tests ----

TEST_CASE(plus_assign_eval)
{
    auto e = lit(3);
    e += lit(4);
    EXPECT(e.eval({}) == value{int64_t{7}});
}

TEST_CASE(minus_assign_eval)
{
    auto e = lit(10);
    e -= lit(3);
    EXPECT(e.eval({}) == value{int64_t{7}});
}

TEST_CASE(times_assign_eval)
{
    auto e = lit(6);
    e *= lit(7);
    EXPECT(e.eval({}) == value{int64_t{42}});
}

TEST_CASE(div_assign_eval)
{
    auto e = lit(10.0);
    e /= lit(4.0);
    EXPECT(e.eval({}) == value{2.5});
}

TEST_CASE(plus_assign_variable)
{
    auto e = var("x");
    e += lit(5);
    EXPECT(e.eval({{"x", int64_t{3}}}) == value{int64_t{8}});
}

TEST_CASE(compound_assign_chain)
{
    auto e = var("x");
    e += lit(1);
    e *= lit(2);
    // (x + 1) * 2 with x=4 → 10
    EXPECT(e.eval({{"x", int64_t{4}}}) == value{int64_t{10}});
}

TEST_CASE(plus_assign_cow)
{
    auto a = lit(3);
    auto b = a;
    EXPECT(a == b);
    b += lit(1);
    // b is now (3 + 1), a is still 3
    EXPECT(a != b);
    EXPECT(a.eval({}) == value{int64_t{3}});
    EXPECT(b.eval({}) == value{int64_t{4}});
}

TEST_CASE(times_assign_cow)
{
    auto x = var("x");
    auto a = x + lit(1);
    auto b = a;
    EXPECT(a == b);
    b *= lit(2);
    // b is now (x+1)*2, a is still x+1
    EXPECT(a != b);
    EXPECT(a.eval({{"x", int64_t{5}}}) == value{int64_t{6}});
    EXPECT(b.eval({{"x", int64_t{5}}}) == value{int64_t{12}});
}

TEST_CASE(compound_assign_cow_shared)
{
    auto x   = var("x");
    auto sub = x + lit(1);
    auto a   = sub;
    auto b   = sub;
    a += lit(10);
    b *= lit(10);
    // a = (x+1)+10, b = (x+1)*10
    EXPECT(a != b);
    EXPECT(a.eval({{"x", int64_t{2}}}) == value{int64_t{13}});
    EXPECT(b.eval({{"x", int64_t{2}}}) == value{int64_t{30}});
    // original sub unchanged
    EXPECT(sub.eval({{"x", int64_t{2}}}) == value{int64_t{3}});
}

TEST_CASE(custom_call_eval)
{
    auto square = call("square", [](auto x) { return x * x; });
    auto x      = var("x");
    auto e      = square(x);
    auto result = e.eval({{"x", int64_t{7}}});
    EXPECT(result == value{int64_t{49}});
}

TEST_CASE(custom_call_interval)
{
    auto square = call("square", [](auto x) { return x * x; });
    auto x      = var("x");
    auto e      = square(x);
    // [2,3] squared: interval*interval = [2,3]*[2,3], products={4,6,6,9}, min=4, max=9
    auto result = e.eval_interval({{"x", interval{int64_t{2}, int64_t{3}}}});
    EXPECT(result == (interval{int64_t{4}, int64_t{9}}));
}

// ---- Math function eval tests ----

TEST_CASE(sin_eval) { EXPECT(sin(lit(0.0)).eval({}) == value{0.0}); }

TEST_CASE(cos_eval) { EXPECT(cos(lit(0.0)).eval({}) == value{1.0}); }

TEST_CASE(tan_eval) { EXPECT(tan(lit(0.0)).eval({}) == value{0.0}); }

TEST_CASE(exp_eval) { EXPECT(exp(lit(0.0)).eval({}) == value{1.0}); }

TEST_CASE(exp_eval_one) { EXPECT(exp(lit(1.0)).eval({}) == value{std::exp(1.0)}); }

TEST_CASE(log_eval) { EXPECT(log(lit(1.0)).eval({}) == value{0.0}); }

TEST_CASE(sqrt_eval_refactored) { EXPECT(sqrt(lit(4.0)).eval({}) == value{2.0}); }

TEST_CASE(abs_int_eval)
{
    EXPECT(abs(lit(-5)).eval({}) == value{int64_t{5}});
    EXPECT(abs(lit(3)).eval({}) == value{int64_t{3}});
}

TEST_CASE(abs_double_eval) { EXPECT(abs(lit(-2.5)).eval({}) == value{2.5}); }

TEST_CASE(floor_eval)
{
    EXPECT(floor(lit(2.7)).eval({}) == value{2.0});
    EXPECT(floor(lit(-2.3)).eval({}) == value{-3.0});
}

TEST_CASE(ceil_eval)
{
    EXPECT(ceil(lit(2.3)).eval({}) == value{3.0});
    EXPECT(ceil(lit(-2.7)).eval({}) == value{-2.0});
}

TEST_CASE(pow_eval) { EXPECT(pow(lit(2.0), lit(3.0)).eval({}) == value{8.0}); }

TEST_CASE(min_eval)
{
    EXPECT(min(lit(3), lit(5)).eval({}) == value{int64_t{3}});
    EXPECT(min(lit(7), lit(2)).eval({}) == value{int64_t{2}});
}

TEST_CASE(max_eval)
{
    EXPECT(max(lit(3), lit(5)).eval({}) == value{int64_t{5}});
    EXPECT(max(lit(7), lit(2)).eval({}) == value{int64_t{7}});
}

TEST_CASE(math_with_variable)
{
    auto x = var("x");
    EXPECT(sin(x).eval({{"x", 0.0}}) == value{0.0});
    EXPECT(abs(x).eval({{"x", int64_t{-7}}}) == value{int64_t{7}});
}

// ---- Interval math function tests ----

TEST_CASE(sin_interval_contains_max)
{
    // sin over [0, π]: reaches max 1.0 at π/2
    const double pi = std::acos(-1.0);
    auto result     = sin(interval{0.0, pi});
    EXPECT(result == (interval{0.0, 1.0}));
}

TEST_CASE(sin_interval_full_period)
{
    const double pi = std::acos(-1.0);
    auto result     = sin(interval{0.0, 2.0 * pi});
    EXPECT(result == (interval{-1.0, 1.0}));
}

TEST_CASE(cos_interval_contains_min)
{
    // cos over [0, π]: reaches min -1.0 at π
    const double pi = std::acos(-1.0);
    auto result     = cos(interval{0.0, pi});
    EXPECT(result == (interval{-1.0, 1.0}));
}

TEST_CASE(cos_interval_monotone)
{
    // cos over [0, 1]: monotonically decreasing
    auto result = cos(interval{0.0, 1.0});
    EXPECT(result == (interval{std::cos(1.0), 1.0}));
}

TEST_CASE(tan_interval_point)
{
    auto result = tan(interval{0.0, 0.0});
    EXPECT(result == (interval{0.0, 0.0}));
}

TEST_CASE(exp_interval)
{
    auto result = exp(interval{0.0, 1.0});
    EXPECT(result == (interval{1.0, std::exp(1.0)}));
}

TEST_CASE(log_interval)
{
    auto result = log(interval{1.0, std::exp(1.0)});
    EXPECT(result == (interval{0.0, 1.0}));
}

TEST_CASE(sqrt_interval_refactored)
{
    auto result = sqrt(interval{4.0, 9.0});
    EXPECT(result == (interval{2.0, 3.0}));
}

TEST_CASE(abs_interval_positive)
{
    auto result = abs(interval{int64_t{2}, int64_t{5}});
    EXPECT(result == (interval{int64_t{2}, int64_t{5}}));
}

TEST_CASE(abs_interval_negative)
{
    auto result = abs(interval{int64_t{-5}, int64_t{-2}});
    EXPECT(result == (interval{int64_t{2}, int64_t{5}}));
}

TEST_CASE(abs_interval_mixed)
{
    auto result = abs(interval{int64_t{-3}, int64_t{5}});
    EXPECT(result == (interval{int64_t{0}, int64_t{5}}));
}

TEST_CASE(abs_interval_mixed_larger_neg)
{
    auto result = abs(interval{int64_t{-7}, int64_t{2}});
    EXPECT(result == (interval{int64_t{0}, int64_t{7}}));
}

TEST_CASE(floor_interval)
{
    auto result = floor(interval{1.2, 3.8});
    EXPECT(result == (interval{1.0, 3.0}));
}

TEST_CASE(ceil_interval)
{
    auto result = ceil(interval{1.2, 3.8});
    EXPECT(result == (interval{2.0, 4.0}));
}

TEST_CASE(pow_interval)
{
    // [2,3]^[2,2] = [4,9]
    auto result = pow(interval{2.0, 3.0}, interval{2.0, 2.0});
    EXPECT(result == (interval{4.0, 9.0}));
}

TEST_CASE(min_interval)
{
    auto result = min(interval{int64_t{1}, int64_t{5}}, interval{int64_t{3}, int64_t{7}});
    EXPECT(result == (interval{int64_t{1}, int64_t{5}}));
}

TEST_CASE(max_interval)
{
    auto result = max(interval{int64_t{1}, int64_t{5}}, interval{int64_t{3}, int64_t{7}});
    EXPECT(result == (interval{int64_t{3}, int64_t{7}}));
}

// ---- Expr math function interval eval tests ----

TEST_CASE(expr_abs_interval)
{
    auto x      = var("x");
    auto result = abs(x).eval_interval({{"x", interval{int64_t{-3}, int64_t{5}}}});
    EXPECT(result == (interval{int64_t{0}, int64_t{5}}));
}

TEST_CASE(expr_min_interval)
{
    auto x      = var("x");
    auto y      = var("y");
    auto result = min(x, y).eval_interval(
        {{"x", interval{int64_t{1}, int64_t{5}}}, {"y", interval{int64_t{3}, int64_t{7}}}});
    EXPECT(result == (interval{int64_t{1}, int64_t{5}}));
}

TEST_CASE(expr_max_interval)
{
    auto x      = var("x");
    auto y      = var("y");
    auto result = max(x, y).eval_interval(
        {{"x", interval{int64_t{1}, int64_t{5}}}, {"y", interval{int64_t{3}, int64_t{7}}}});
    EXPECT(result == (interval{int64_t{3}, int64_t{7}}));
}

TEST_CASE(expr_exp_interval)
{
    auto x      = var("x");
    auto result = exp(x).eval_interval({{"x", interval{0.0, 1.0}}});
    EXPECT(result == (interval{1.0, std::exp(1.0)}));
}

// ---- to_string tests ----

TEST_CASE(to_string_literal_int)
{
    EXPECT(lit(42).to_string() == "42");
    EXPECT(lit(-7).to_string() == "-7");
}

TEST_CASE(to_string_literal_double)
{
    EXPECT(lit(3.14).to_string() == "3.14");
    EXPECT(lit(0.0).to_string() == "0");
}

TEST_CASE(to_string_variable) { EXPECT(var("x").to_string() == "x"); }

TEST_CASE(to_string_add)
{
    auto x = var("x");
    EXPECT((x + lit(3)).to_string() == "(x + 3)");
}

TEST_CASE(to_string_sub)
{
    auto x = var("x");
    EXPECT((x - lit(1)).to_string() == "(x - 1)");
}

TEST_CASE(to_string_mul)
{
    auto x = var("x");
    EXPECT((x * lit(2)).to_string() == "(x * 2)");
}

TEST_CASE(to_string_div)
{
    auto x = var("x");
    EXPECT((x / lit(4)).to_string() == "(x / 4)");
}

TEST_CASE(to_string_neg)
{
    auto x = var("x");
    EXPECT((-x).to_string() == "(-x)");
}

TEST_CASE(to_string_nested)
{
    auto x = var("x");
    auto y = var("y");
    auto e = (x + lit(1)) * (y - lit(2));
    EXPECT(e.to_string() == "((x + 1) * (y - 2))");
}

TEST_CASE(to_string_function)
{
    auto x = var("x");
    EXPECT(sin(x).to_string() == "sin(x)");
    EXPECT(sqrt(x).to_string() == "sqrt(x)");
    EXPECT(abs(x).to_string() == "abs(x)");
}

TEST_CASE(to_string_function_two_arg)
{
    auto x = var("x");
    auto y = var("y");
    EXPECT(pow(x, y).to_string() == "pow(x, y)");
    EXPECT(min(x, y).to_string() == "min(x, y)");
    EXPECT(max(x, y).to_string() == "max(x, y)");
}

TEST_CASE(to_string_composed)
{
    auto x = var("x");
    auto e = sin(x * lit(2)) + lit(1);
    EXPECT(e.to_string() == "(sin((x * 2)) + 1)");
}

TEST_CASE(free_to_string)
{
    auto x = var("x");
    EXPECT(to_string(x + lit(1)) == "(x + 1)");
    EXPECT(to_string(sin(x)) == "sin(x)");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
