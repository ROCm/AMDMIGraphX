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
#include <migraphx/serialize.hpp>
#include <sstream>
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
using migraphx::sym::parse;
using migraphx::sym::pow;
using migraphx::sym::pvar;
using migraphx::sym::rewrite_rule;
using migraphx::sym::scalar;
using migraphx::sym::simplify;
using migraphx::sym::sin;
using migraphx::sym::sqrt;
using migraphx::sym::tan;
using migraphx::sym::to_string;
using migraphx::sym::var;

// ---- Value evaluation tests ----

TEST_CASE(literal_int_eval)
{
    auto e      = lit(42);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{42}});
}

TEST_CASE(literal_double_eval)
{
    auto e      = lit(3.14);
    auto result = e.eval({});
    EXPECT(result == scalar{3.14});
}

TEST_CASE(variable_eval)
{
    auto e      = var("x");
    auto result = e.eval({{"x", int64_t{10}}});
    EXPECT(result == scalar{int64_t{10}});
}

TEST_CASE(add_int_eval)
{
    auto e      = lit(3) + lit(4);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{7}});
}

TEST_CASE(add_mixed_eval)
{
    auto e      = lit(3) + lit(1.5);
    auto result = e.eval({});
    EXPECT(result == scalar{4.5});
}

TEST_CASE(sub_eval)
{
    auto e      = lit(10) - lit(3);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{7}});
}

TEST_CASE(mul_eval)
{
    auto e      = lit(6) * lit(7);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{42}});
}

TEST_CASE(div_double_eval)
{
    auto e      = lit(10.0) / lit(4.0);
    auto result = e.eval({});
    EXPECT(result == scalar{2.5});
}

TEST_CASE(mod_int_eval)
{
    auto e      = lit(10) % lit(3);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{1}});
}

TEST_CASE(mod_double_eval)
{
    auto e      = lit(10.5) % lit(3.0);
    auto result = e.eval({});
    EXPECT(result == scalar{std::fmod(10.5, 3.0)});
}

TEST_CASE(mod_variable_eval)
{
    auto e      = var("x") % lit(3);
    auto result = e.eval({{"x", int64_t{10}}});
    EXPECT(result == scalar{int64_t{1}});
}

TEST_CASE(neg_eval)
{
    auto e      = -lit(5);
    auto result = e.eval({});
    EXPECT(result == scalar{int64_t{-5}});
}

TEST_CASE(compound_expr_eval)
{
    auto x      = var("x");
    auto e      = (x + lit(3)) * lit(2);
    auto result = e.eval({{"x", int64_t{5}}});
    EXPECT(result == scalar{int64_t{16}});
}

TEST_CASE(multi_variable_eval)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x * y + lit(1);

    auto result = e.eval({{"x", int64_t{3}}, {"y", int64_t{4}}});
    EXPECT(result == scalar{int64_t{13}});
}

TEST_CASE(sqrt_eval)
{
    auto e      = sqrt(lit(4.0));
    auto result = e.eval({});
    EXPECT(result == scalar{2.0});
}

TEST_CASE(sqrt_int_eval)
{
    auto e      = sqrt(lit(9));
    auto result = e.eval({});
    EXPECT(result == scalar{3.0});
}

TEST_CASE(nested_sqrt_eval)
{
    auto e      = sqrt(lit(16.0)) + lit(1.0);
    auto result = e.eval({});
    EXPECT(result == scalar{5.0});
}

TEST_CASE(arg_int_literal)
{
    auto x      = var("x");
    auto e      = call("+", [](auto a, auto b) { return a + b; })(x, 3);
    auto result = e.eval({{"x", int64_t{5}}});
    EXPECT(result == scalar{int64_t{8}});
}

TEST_CASE(arg_double_literal)
{
    auto x      = var("x");
    auto e      = call("*", [](auto a, auto b) { return a * b; })(x, 2.0);
    auto result = e.eval({{"x", 3.0}});
    EXPECT(result == scalar{6.0});
}

TEST_CASE(shared_subexpr)
{
    auto x      = var("x");
    auto sub    = x + lit(1);
    auto e      = sub * sub;
    auto result = e.eval({{"x", int64_t{4}}});
    EXPECT(result == scalar{int64_t{25}});
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

TEST_CASE(mod_interval)
{
    auto x = var("x");
    auto e = x % lit(3);
    // [7,10] % 3: 7%3=1, 10%3=1 → min=1, max=1 (endpoint case)
    auto result = e.eval_interval({{"x", interval{int64_t{7}, int64_t{10}}}});
    EXPECT(result == (interval{int64_t{1}, int64_t{1}}));
}

TEST_CASE(mod_interval_range)
{
    auto x = var("x");
    auto e = x % lit(5);
    // [3,8] % 5: 3%5=3, 8%5=3 → min=3, max=3
    auto result = e.eval_interval({{"x", interval{int64_t{3}, int64_t{8}}}});
    EXPECT(result == (interval{int64_t{3}, int64_t{3}}));
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

TEST_CASE(interval_mod_assign)
{
    interval a{int64_t{7}, int64_t{10}};
    a %= interval{int64_t{3}, int64_t{3}};
    EXPECT(a == (interval{int64_t{1}, int64_t{1}}));
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

// ---- empty tests ----

TEST_CASE(empty_default)
{
    expr e;
    EXPECT(e.empty());
}

TEST_CASE(empty_literal)
{
    auto e = lit(42);
    EXPECT(not e.empty());
}

TEST_CASE(empty_variable)
{
    auto e = var("x");
    EXPECT(not e.empty());
}

TEST_CASE(empty_compound)
{
    auto e = var("x") + lit(1);
    EXPECT(not e.empty());
}

// ---- hash tests ----

TEST_CASE(hash_equal_exprs)
{
    auto a = var("x") + lit(1);
    auto b = var("x") + lit(1);
    EXPECT(a.hash() == b.hash());
}

TEST_CASE(hash_different_exprs)
{
    auto a = var("x") + lit(1);
    auto b = var("x") + lit(2);
    EXPECT(a.hash() != b.hash());
}

TEST_CASE(hash_default_expr)
{
    expr e;
    EXPECT(e.hash() == 0);
}

TEST_CASE(hash_literal)
{
    auto a = lit(42);
    auto b = lit(42);
    EXPECT(a.hash() == b.hash());
}

TEST_CASE(hash_different_literals) { EXPECT(lit(1).hash() != lit(2).hash()); }

TEST_CASE(hash_different_variables) { EXPECT(var("x").hash() != var("y").hash()); }

TEST_CASE(hash_unordered_map_key)
{
    std::unordered_map<expr, int> m;
    auto x = var("x");
    auto y = var("y");
    m[x]   = 10;
    m[y]   = 20;
    EXPECT(m.at(x) == 10);
    EXPECT(m.at(y) == 20);
}

// ---- eval_uint tests ----

TEST_CASE(eval_uint_literal)
{
    auto e = lit(42);
    EXPECT(e.eval_uint({}) == 42);
}

TEST_CASE(eval_uint_compound)
{
    auto e = lit(3) + lit(4);
    EXPECT(e.eval_uint({}) == 7);
}

TEST_CASE(eval_uint_symbol_map)
{
    auto x = var("x");
    EXPECT(x.eval_uint({{x, 10}}) == 10);
}

TEST_CASE(eval_uint_symbol_map_compound)
{
    auto x = var("x");
    auto e = x + lit(5);
    EXPECT(e.eval_uint({{e, 42}}) == 42);
}

TEST_CASE(eval_uint_symbol_map_partial)
{
    auto x = var("x");
    auto e = x * lit(2);
    // Map x to 7, so x*2 = 14
    auto inner = lit(7) * lit(2);
    EXPECT(inner.eval_uint({}) == 14);
}

// ---- subs tests ----

TEST_CASE(subs_variable)
{
    auto x = var("x");
    auto e = x.subs({{x, lit(42)}});
    EXPECT(e == lit(42));
}

TEST_CASE(subs_compound)
{
    auto x = var("x");
    auto e = (x + lit(1)).subs({{x, lit(5)}});
    EXPECT(e.eval({}) == scalar{int64_t{6}});
}

TEST_CASE(subs_no_match)
{
    auto x = var("x");
    auto y = var("y");
    auto e = x.subs({{y, lit(5)}});
    EXPECT(e == x);
}

TEST_CASE(subs_nested)
{
    auto x = var("x");
    auto y = var("y");
    auto e = (x + y).subs({{x, lit(3)}, {y, lit(4)}});
    EXPECT(e.eval({}) == scalar{int64_t{7}});
}

TEST_CASE(subs_subexpr)
{
    auto x      = var("x");
    auto sub    = x + lit(1);
    auto e      = sin(sub);
    auto result = e.subs({{sub, lit(0)}});
    // sin(0) = 0.0
    EXPECT(result.eval({}) == scalar{0.0});
}

TEST_CASE(subs_literal_unchanged)
{
    auto e = lit(42).subs({{var("x"), lit(5)}});
    EXPECT(e == lit(42));
}

TEST_CASE(subs_empty_map)
{
    auto x = var("x");
    auto e = x.subs({});
    EXPECT(e == x);
}

TEST_CASE(subs_default_expr)
{
    expr e;
    auto result = e.subs({{var("x"), lit(1)}});
    EXPECT(result.empty());
}

// ---- Compound assignment tests ----

TEST_CASE(plus_assign_eval)
{
    auto e = lit(3);
    e += lit(4);
    EXPECT(e.eval({}) == scalar{int64_t{7}});
}

TEST_CASE(minus_assign_eval)
{
    auto e = lit(10);
    e -= lit(3);
    EXPECT(e.eval({}) == scalar{int64_t{7}});
}

TEST_CASE(times_assign_eval)
{
    auto e = lit(6);
    e *= lit(7);
    EXPECT(e.eval({}) == scalar{int64_t{42}});
}

TEST_CASE(div_assign_eval)
{
    auto e = lit(10.0);
    e /= lit(4.0);
    EXPECT(e.eval({}) == scalar{2.5});
}

TEST_CASE(plus_assign_variable)
{
    auto e = var("x");
    e += lit(5);
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{8}});
}

TEST_CASE(compound_assign_chain)
{
    auto e = var("x");
    e += lit(1);
    e *= lit(2);
    // (x + 1) * 2 with x=4 → 10
    EXPECT(e.eval({{"x", int64_t{4}}}) == scalar{int64_t{10}});
}

TEST_CASE(plus_assign_cow)
{
    auto a = lit(3);
    auto b = a;
    EXPECT(a == b);
    b += lit(1);
    // b is now (3 + 1), a is still 3
    EXPECT(a != b);
    EXPECT(a.eval({}) == scalar{int64_t{3}});
    EXPECT(b.eval({}) == scalar{int64_t{4}});
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
    EXPECT(a.eval({{"x", int64_t{5}}}) == scalar{int64_t{6}});
    EXPECT(b.eval({{"x", int64_t{5}}}) == scalar{int64_t{12}});
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
    EXPECT(a.eval({{"x", int64_t{2}}}) == scalar{int64_t{13}});
    EXPECT(b.eval({{"x", int64_t{2}}}) == scalar{int64_t{30}});
    // original sub unchanged
    EXPECT(sub.eval({{"x", int64_t{2}}}) == scalar{int64_t{3}});
}

// ---- Non-expr operator tests (int64_t / double mixed with expr) ----

TEST_CASE(add_expr_int64)
{
    auto e = var("x") + int64_t{5};
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{8}});
}

TEST_CASE(add_int64_expr)
{
    auto e = int64_t{5} + var("x");
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{8}});
}

TEST_CASE(add_expr_double)
{
    auto e = var("x") + 1.5;
    EXPECT(e.eval({{"x", int64_t{2}}}) == scalar{3.5});
}

TEST_CASE(add_double_expr)
{
    auto e = 1.5 + var("x");
    EXPECT(e.eval({{"x", int64_t{2}}}) == scalar{3.5});
}

TEST_CASE(sub_expr_int64)
{
    auto e = var("x") - int64_t{3};
    EXPECT(e.eval({{"x", int64_t{10}}}) == scalar{int64_t{7}});
}

TEST_CASE(sub_int64_expr)
{
    auto e = int64_t{10} - var("x");
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{7}});
}

TEST_CASE(sub_expr_double)
{
    auto e = var("x") - 0.5;
    EXPECT(e.eval({{"x", 2.0}}) == scalar{1.5});
}

TEST_CASE(sub_double_expr)
{
    auto e = 10.0 - var("x");
    EXPECT(e.eval({{"x", 3.0}}) == scalar{7.0});
}

TEST_CASE(mul_expr_int64)
{
    auto e = var("x") * int64_t{6};
    EXPECT(e.eval({{"x", int64_t{7}}}) == scalar{int64_t{42}});
}

TEST_CASE(mul_int64_expr)
{
    auto e = int64_t{6} * var("x");
    EXPECT(e.eval({{"x", int64_t{7}}}) == scalar{int64_t{42}});
}

TEST_CASE(mul_expr_double)
{
    auto e = var("x") * 2.5;
    EXPECT(e.eval({{"x", 4.0}}) == scalar{10.0});
}

TEST_CASE(mul_double_expr)
{
    auto e = 2.5 * var("x");
    EXPECT(e.eval({{"x", 4.0}}) == scalar{10.0});
}

TEST_CASE(div_expr_int64)
{
    auto e = var("x") / int64_t{2};
    EXPECT(e.eval({{"x", 10.0}}) == scalar{5.0});
}

TEST_CASE(div_int64_expr)
{
    auto e = int64_t{10} / var("x");
    EXPECT(e.eval({{"x", 2.0}}) == scalar{5.0});
}

TEST_CASE(div_expr_double)
{
    auto e = var("x") / 4.0;
    EXPECT(e.eval({{"x", 10.0}}) == scalar{2.5});
}

TEST_CASE(div_double_expr)
{
    auto e = 10.0 / var("x");
    EXPECT(e.eval({{"x", 4.0}}) == scalar{2.5});
}

TEST_CASE(plus_assign_int64)
{
    auto e = var("x");
    e += int64_t{5};
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{8}});
}

TEST_CASE(plus_assign_double)
{
    auto e = var("x");
    e += 1.5;
    EXPECT(e.eval({{"x", 2.0}}) == scalar{3.5});
}

TEST_CASE(minus_assign_int64)
{
    auto e = var("x");
    e -= int64_t{3};
    EXPECT(e.eval({{"x", int64_t{10}}}) == scalar{int64_t{7}});
}

TEST_CASE(minus_assign_double)
{
    auto e = var("x");
    e -= 0.5;
    EXPECT(e.eval({{"x", 2.0}}) == scalar{1.5});
}

TEST_CASE(times_assign_int64)
{
    auto e = var("x");
    e *= int64_t{6};
    EXPECT(e.eval({{"x", int64_t{7}}}) == scalar{int64_t{42}});
}

TEST_CASE(times_assign_double)
{
    auto e = var("x");
    e *= 2.5;
    EXPECT(e.eval({{"x", 4.0}}) == scalar{10.0});
}

TEST_CASE(div_assign_int64)
{
    auto e = var("x");
    e /= int64_t{2};
    EXPECT(e.eval({{"x", 10.0}}) == scalar{5.0});
}

TEST_CASE(div_assign_double)
{
    auto e = var("x");
    e /= 4.0;
    EXPECT(e.eval({{"x", 10.0}}) == scalar{2.5});
}

TEST_CASE(mod_expr_int64)
{
    auto e = var("x") % int64_t{3};
    EXPECT(e.eval({{"x", int64_t{10}}}) == scalar{int64_t{1}});
}

TEST_CASE(mod_int64_expr)
{
    auto e = int64_t{10} % var("x");
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{1}});
}

TEST_CASE(mod_expr_double)
{
    auto e = var("x") % 3.0;
    EXPECT(e.eval({{"x", 10.5}}) == scalar{std::fmod(10.5, 3.0)});
}

TEST_CASE(mod_double_expr)
{
    auto e = 10.5 % var("x");
    EXPECT(e.eval({{"x", 3.0}}) == scalar{std::fmod(10.5, 3.0)});
}

TEST_CASE(mod_assign_expr)
{
    auto e = var("x");
    e %= lit(3);
    EXPECT(e.eval({{"x", int64_t{10}}}) == scalar{int64_t{1}});
}

TEST_CASE(mod_assign_int64)
{
    auto e = var("x");
    e %= int64_t{3};
    EXPECT(e.eval({{"x", int64_t{10}}}) == scalar{int64_t{1}});
}

TEST_CASE(mod_assign_double)
{
    auto e = var("x");
    e %= 3.0;
    EXPECT(e.eval({{"x", 10.5}}) == scalar{std::fmod(10.5, 3.0)});
}

TEST_CASE(non_expr_compound)
{
    // (x + 3) * 2.0 - 1  with x=4 → (7) * 2.0 - 1 = 13.0
    auto e = (var("x") + int64_t{3}) * 2.0 - int64_t{1};
    EXPECT(e.eval({{"x", int64_t{4}}}) == scalar{13.0});
}

TEST_CASE(non_expr_both_sides)
{
    // 2 * x + 1.5  with x=3 → 7.5
    auto e = int64_t{2} * var("x") + 1.5;
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{7.5});
}

TEST_CASE(non_expr_chain_assign)
{
    auto e = var("x");
    e += int64_t{1};
    e *= 2.0;
    e -= int64_t{1};
    // (x + 1) * 2.0 - 1  with x=4 → 9.0
    EXPECT(e.eval({{"x", int64_t{4}}}) == scalar{9.0});
}

TEST_CASE(custom_call_eval)
{
    auto square = call("square", [](auto x) { return x * x; });
    auto x      = var("x");
    auto e      = square(x);
    auto result = e.eval({{"x", int64_t{7}}});
    EXPECT(result == scalar{int64_t{49}});
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

TEST_CASE(sin_eval) { EXPECT(sin(lit(0.0)).eval({}) == scalar{0.0}); }

TEST_CASE(cos_eval) { EXPECT(cos(lit(0.0)).eval({}) == scalar{1.0}); }

TEST_CASE(tan_eval) { EXPECT(tan(lit(0.0)).eval({}) == scalar{0.0}); }

TEST_CASE(exp_eval) { EXPECT(exp(lit(0.0)).eval({}) == scalar{1.0}); }

TEST_CASE(exp_eval_one) { EXPECT(exp(lit(1.0)).eval({}) == scalar{std::exp(1.0)}); }

TEST_CASE(log_eval) { EXPECT(log(lit(1.0)).eval({}) == scalar{0.0}); }

TEST_CASE(sqrt_eval_refactored) { EXPECT(sqrt(lit(4.0)).eval({}) == scalar{2.0}); }

TEST_CASE(abs_int_eval)
{
    EXPECT(abs(lit(-5)).eval({}) == scalar{int64_t{5}});
    EXPECT(abs(lit(3)).eval({}) == scalar{int64_t{3}});
}

TEST_CASE(abs_double_eval) { EXPECT(abs(lit(-2.5)).eval({}) == scalar{2.5}); }

TEST_CASE(floor_eval)
{
    EXPECT(floor(lit(2.7)).eval({}) == scalar{2.0});
    EXPECT(floor(lit(-2.3)).eval({}) == scalar{-3.0});
}

TEST_CASE(ceil_eval)
{
    EXPECT(ceil(lit(2.3)).eval({}) == scalar{3.0});
    EXPECT(ceil(lit(-2.7)).eval({}) == scalar{-2.0});
}

TEST_CASE(pow_eval) { EXPECT(pow(lit(2.0), lit(3.0)).eval({}) == scalar{8.0}); }

TEST_CASE(min_eval)
{
    EXPECT(min(lit(3), lit(5)).eval({}) == scalar{int64_t{3}});
    EXPECT(min(lit(7), lit(2)).eval({}) == scalar{int64_t{2}});
}

TEST_CASE(max_eval)
{
    EXPECT(max(lit(3), lit(5)).eval({}) == scalar{int64_t{5}});
    EXPECT(max(lit(7), lit(2)).eval({}) == scalar{int64_t{7}});
}

TEST_CASE(math_with_variable)
{
    auto x = var("x");
    EXPECT(sin(x).eval({{"x", 0.0}}) == scalar{0.0});
    EXPECT(abs(x).eval({{"x", int64_t{-7}}}) == scalar{int64_t{7}});
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
    // variables sort before literals
    EXPECT((x + lit(3)).to_string() == "x + 3");
}

TEST_CASE(to_string_sub)
{
    auto x = var("x");
    // x - 1 is rewritten as x + (-1), variables sort before literals
    EXPECT((x - lit(1)).to_string() == "x + -1");
}

TEST_CASE(to_string_mul)
{
    auto x = var("x");
    // literals sort before variables in multiplication
    EXPECT((x * lit(2)).to_string() == "2*x");
}

TEST_CASE(to_string_div)
{
    auto x = var("x");
    EXPECT((x / lit(4)).to_string() == "x/4");
}

TEST_CASE(to_string_mod)
{
    auto x = var("x");
    EXPECT((x % lit(3)).to_string() == "x%3");
}

TEST_CASE(to_string_neg)
{
    auto x = var("x");
    // -x is rewritten as -1 * x
    EXPECT((-x).to_string() == "-1*x");
}

TEST_CASE(to_string_nested)
{
    auto x = var("x");
    auto y = var("y");
    auto e = (x + lit(1)) * (y - lit(2));
    // fully expanded: (x+1)*(y-2) = xy - 2x + y - 2
    // ops first, then variables, then literals
    EXPECT(e.to_string() == "x*y + -2*x + y + -2");
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
    // lit(1) sorts before sin(...)
    EXPECT(e.to_string() == "sin(2*x) + 1");
}

TEST_CASE(free_to_string)
{
    auto x = var("x");
    EXPECT(to_string(x + lit(1)) == "x + 1");
    EXPECT(to_string(sin(x)) == "sin(x)");
}

// ---- ostream operator<< tests ----

TEST_CASE(ostream_expr)
{
    std::ostringstream ss;
    ss << (var("x") + lit(1));
    EXPECT(ss.str() == "x + 1");
}

TEST_CASE(ostream_expr_function)
{
    std::ostringstream ss;
    ss << sin(var("x"));
    EXPECT(ss.str() == "sin(x)");
}

TEST_CASE(ostream_interval)
{
    std::ostringstream ss;
    ss << interval{int64_t{1}, int64_t{10}};
    EXPECT(ss.str() == "[1, 10]");
}

TEST_CASE(ostream_interval_double)
{
    std::ostringstream ss;
    ss << interval{1.5, 3.5};
    EXPECT(ss.str() == "[1.5, 3.5]");
}

// ---- Associative flattening tests ----

TEST_CASE(flatten_add_right)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a + b) + c should flatten to +(a, b, c)
    auto e      = (a + b) + c;
    auto result = e.eval({{"a", int64_t{1}}, {"b", int64_t{2}}, {"c", int64_t{3}}});
    EXPECT(result == scalar{int64_t{6}});
    EXPECT(e.children().size() == 3);
}

TEST_CASE(flatten_add_left)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // a + (b + c) should flatten to +(a, b, c)
    auto e      = a + (b + c);
    auto result = e.eval({{"a", int64_t{1}}, {"b", int64_t{2}}, {"c", int64_t{3}}});
    EXPECT(result == scalar{int64_t{6}});
    EXPECT(e.children().size() == 3);
}

TEST_CASE(flatten_add_both)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    // (a + b) + (c + d) should flatten to +(a, b, c, d)
    auto e = (a + b) + (c + d);
    auto result =
        e.eval({{"a", int64_t{1}}, {"b", int64_t{2}}, {"c", int64_t{3}}, {"d", int64_t{4}}});
    EXPECT(result == scalar{int64_t{10}});
    EXPECT(e.children().size() == 4);
}

TEST_CASE(flatten_mul_right)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a * b) * c should flatten to *(a, b, c)
    auto e      = (a * b) * c;
    auto result = e.eval({{"a", int64_t{2}}, {"b", int64_t{3}}, {"c", int64_t{4}}});
    EXPECT(result == scalar{int64_t{24}});
    EXPECT(e.children().size() == 3);
}

TEST_CASE(flatten_mul_both)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    // (a * b) * (c * d) should flatten to *(a, b, c, d)
    auto e = (a * b) * (c * d);
    auto result =
        e.eval({{"a", int64_t{2}}, {"b", int64_t{3}}, {"c", int64_t{4}}, {"d", int64_t{5}}});
    EXPECT(result == scalar{int64_t{120}});
    EXPECT(e.children().size() == 4);
}

TEST_CASE(flatten_nested_add)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    // ((a + b) + c) + d should flatten to +(a, b, c, d)
    auto e = ((a + b) + c) + d;
    auto result =
        e.eval({{"a", int64_t{1}}, {"b", int64_t{2}}, {"c", int64_t{3}}, {"d", int64_t{4}}});
    EXPECT(result == scalar{int64_t{10}});
    EXPECT(e.children().size() == 4);
}

TEST_CASE(sub_flattens_into_add)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a - b) - c becomes a + (-b) + (-c), flattened to 3 children
    auto e      = (a - b) - c;
    auto result = e.eval({{"a", int64_t{10}}, {"b", int64_t{3}}, {"c", int64_t{2}}});
    EXPECT(result == scalar{int64_t{5}});
    EXPECT(e.children().size() == 3);
}

TEST_CASE(no_flatten_div)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a / b) / c should NOT flatten
    auto e      = (a / b) / c;
    auto result = e.eval({{"a", 12.0}, {"b", 2.0}, {"c", 3.0}});
    EXPECT(result == scalar{2.0});
    EXPECT(e.children().size() == 2);
}

TEST_CASE(no_flatten_mixed_ops)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a * b) + c should NOT flatten mul into add
    auto e = (a * b) + c;
    EXPECT(e.children().size() == 2);
    auto result = e.eval({{"a", int64_t{3}}, {"b", int64_t{4}}, {"c", int64_t{5}}});
    EXPECT(result == scalar{int64_t{17}});
}

TEST_CASE(flatten_add_interval)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto e = (a + b) + c;
    // [1,2] + [3,4] + [5,6] = [9,12]
    auto result = e.eval_interval({{"a", interval{int64_t{1}, int64_t{2}}},
                                   {"b", interval{int64_t{3}, int64_t{4}}},
                                   {"c", interval{int64_t{5}, int64_t{6}}}});
    EXPECT(result == (interval{int64_t{9}, int64_t{12}}));
}

TEST_CASE(flatten_mul_interval)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto e = (a * b) * c;
    // [1,2] * [3,4] * [1,1] = [3,8] * [1,1] = [3,8]
    auto result = e.eval_interval({{"a", interval{int64_t{1}, int64_t{2}}},
                                   {"b", interval{int64_t{3}, int64_t{4}}},
                                   {"c", interval{int64_t{1}, int64_t{1}}}});
    EXPECT(result == (interval{int64_t{3}, int64_t{8}}));
}

TEST_CASE(flatten_to_string_add)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    EXPECT(((a + b) + c).to_string() == "c + b + a");
}

TEST_CASE(flatten_to_string_add_both)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    EXPECT(((a + b) + (c + d)).to_string() == "d + c + b + a");
}

TEST_CASE(flatten_to_string_mul)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    EXPECT(((a * b) * c).to_string() == "a*b*c");
}

TEST_CASE(flatten_to_string_nested)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    EXPECT((((a + b) + c) + d).to_string() == "d + c + b + a");
}

TEST_CASE(flatten_to_string_mixed)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // (a * b) + c: mul is a child of add, should not flatten across ops
    // ops sort before variables
    EXPECT(((a * b) + c).to_string() == "a*b + c");
}

// ---- Constant folding tests ----

TEST_CASE(const_fold_add)
{
    auto e = lit(3) + lit(4);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{7}});
}

TEST_CASE(const_fold_sub)
{
    auto e = lit(10) - lit(3);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{7}});
}

TEST_CASE(const_fold_mul)
{
    auto e = lit(6) * lit(7);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{42}});
}

TEST_CASE(const_fold_div)
{
    auto e = lit(10.0) / lit(4.0);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{2.5});
}

TEST_CASE(const_fold_neg)
{
    auto e = -lit(5);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{-5}});
}

TEST_CASE(const_fold_nested)
{
    // (3 + 4) * 2 should fold completely to 14
    auto e = (lit(3) + lit(4)) * lit(2);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{14}});
}

TEST_CASE(const_fold_math_functions)
{
    EXPECT(sin(lit(0.0)).name() == "literal");
    EXPECT(cos(lit(0.0)).name() == "literal");
    EXPECT(sqrt(lit(4.0)).name() == "literal");
    EXPECT(abs(lit(-5)).name() == "literal");
    EXPECT(floor(lit(2.7)).name() == "literal");
    EXPECT(ceil(lit(2.3)).name() == "literal");
    EXPECT(pow(lit(2.0), lit(3.0)).name() == "literal");
    EXPECT(min(lit(3), lit(5)).name() == "literal");
    EXPECT(max(lit(3), lit(5)).name() == "literal");
}

TEST_CASE(no_const_fold_with_variable)
{
    auto x = var("x");
    auto e = x + lit(3);
    EXPECT(e.name() != "literal");
}

TEST_CASE(const_fold_partial)
{
    auto x = var("x");
    // x + (3 + 4): the (3+4) subexpr folds to 7, but x+7 does not fold
    auto e = x + (lit(3) + lit(4));
    EXPECT(e.name() != "literal");
    auto result = e.eval({{"x", int64_t{1}}});
    EXPECT(result == scalar{int64_t{8}});
}

TEST_CASE(const_fold_chain)
{
    // lit(1) + lit(2) + lit(3) should flatten then fold to 6
    auto e = lit(1) + lit(2) + lit(3);
    EXPECT(e.name() == "literal");
    EXPECT(e.eval({}) == scalar{int64_t{6}});
}

TEST_CASE(const_fold_to_string)
{
    auto e = lit(3) + lit(4);
    EXPECT(e.to_string() == "7");
}

// ---- Associative constant folding tests ----

TEST_CASE(assoc_fold_add_trailing_literals)
{
    auto x = var("x");
    // x + 2 + 3: flattened to +(x, 2, 3), adjacent literals 2 and 3 fold to 5
    auto e = x + lit(2) + lit(3);
    EXPECT(e.eval({{"x", int64_t{0}}}) == scalar{int64_t{5}});
    EXPECT(e == x + lit(5));
}

TEST_CASE(assoc_fold_mul_trailing_literals)
{
    auto x = var("x");
    // x * 2 * 3: flattened to *(x, 2, 3), adjacent literals 2 and 3 fold to 6
    auto e = x * lit(2) * lit(3);
    EXPECT(e.eval({{"x", int64_t{1}}}) == scalar{int64_t{6}});
    EXPECT(e == x * lit(6));
}

TEST_CASE(assoc_fold_add_leading_literals)
{
    auto x = var("x");
    // 2 + 3 + x: literals adjacent at the front fold to 5
    auto e = lit(2) + lit(3) + x;
    EXPECT(e == x + lit(5));
}

TEST_CASE(assoc_fold_mul_leading_literals)
{
    auto x = var("x");
    // 2 * 3 * x: literals adjacent at the front fold to 6
    auto e = lit(2) * lit(3) * x;
    EXPECT(e == lit(6) * x);
}

TEST_CASE(assoc_fold_add_three_literals)
{
    // 1 + 2 + 3: all literals fold completely
    auto e = lit(1) + lit(2) + lit(3);
    EXPECT(e == lit(6));
}

TEST_CASE(assoc_fold_mul_three_literals)
{
    // 2 * 3 * 4: all literals fold completely
    auto e = lit(2) * lit(3) * lit(4);
    EXPECT(e == lit(24));
}

TEST_CASE(assoc_fold_add_mixed_chain)
{
    auto x = var("x");
    auto y = var("y");
    // x + 1 + y + 2: after sorting, literals end up adjacent and fold
    auto e = x + lit(1) + y + lit(2);
    EXPECT(e.eval({{"x", int64_t{10}}, {"y", int64_t{20}}}) == scalar{int64_t{33}});
}

TEST_CASE(assoc_fold_mul_mixed_chain)
{
    auto x = var("x");
    auto y = var("y");
    // x * 2 * y * 3: after sorting, literals end up adjacent and fold
    auto e = x * lit(2) * y * lit(3);
    EXPECT(e.eval({{"x", int64_t{5}}, {"y", int64_t{7}}}) == scalar{int64_t{210}});
}

TEST_CASE(assoc_fold_preserves_eval)
{
    auto x = var("x");
    // Folding must not change evaluation results
    auto e1 = x + lit(10) + lit(20) + lit(30);
    auto e2 = x + lit(60);
    EXPECT(e1.eval({{"x", int64_t{5}}}) == e2.eval({{"x", int64_t{5}}}));
    EXPECT(e1 == e2);
}

TEST_CASE(assoc_fold_double_literals)
{
    auto x = var("x");
    // Folding works with double literals too
    auto e = x + lit(1.5) + lit(2.5);
    EXPECT(e.eval({{"x", 0.0}}) == scalar{4.0});
}

TEST_CASE(assoc_fold_no_fold_single_literal)
{
    auto x = var("x");
    // Only one literal, nothing to fold
    auto e = x + lit(5);
    EXPECT(e.eval({{"x", int64_t{3}}}) == scalar{int64_t{8}});
}

// ---- Canonicalization tests ----

TEST_CASE(canonical_add_commutative)
{
    auto x = var("x");
    auto y = var("y");
    // x + y and y + x should be the same expression
    EXPECT(x + y == y + x);
}

TEST_CASE(canonical_mul_commutative)
{
    auto x = var("x");
    auto y = var("y");
    // x * y and y * x should be the same expression
    EXPECT(x * y == y * x);
}

TEST_CASE(canonical_add_three_vars)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // all orderings should produce the same expression
    EXPECT((a + b) + c == (c + a) + b);
    EXPECT((a + b) + c == (b + c) + a);
}

TEST_CASE(canonical_mul_three_vars)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    EXPECT((a * b) * c == (c * a) * b);
    EXPECT((a * b) * c == (b * c) * a);
}

TEST_CASE(canonical_add_lit_var_order)
{
    auto x = var("x");
    // lit + var and var + lit should be the same
    EXPECT(lit(5) + x == x + lit(5));
}

TEST_CASE(canonical_mul_lit_var_order)
{
    auto x = var("x");
    EXPECT(lit(3) * x == x * lit(3));
}

TEST_CASE(canonical_div_not_commutative)
{
    auto x = var("x");
    auto y = var("y");
    // division is not commutative, order should be preserved
    EXPECT(x / y != y / x);
}

TEST_CASE(canonical_compound_commutative)
{
    auto x = var("x");
    auto y = var("y");
    // (x+1) * (y+2) and (y+2) * (x+1) should be the same
    EXPECT((x + lit(1)) * (y + lit(2)) == (y + lit(2)) * (x + lit(1)));
}

TEST_CASE(canonical_nested_commutative)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // a + b*c and b*c + a should be the same
    EXPECT(a + b * c == b * c + a);
}

TEST_CASE(canonical_eval_preserved)
{
    auto x = var("x");
    auto y = var("y");
    // canonicalization should not change evaluation results
    auto e1 = x + y;
    auto e2 = y + x;
    EXPECT(e1.eval({{"x", int64_t{3}}, {"y", int64_t{7}}}) == scalar{int64_t{10}});
    EXPECT(e2.eval({{"x", int64_t{3}}, {"y", int64_t{7}}}) == scalar{int64_t{10}});

    auto e3 = x * y;
    auto e4 = y * x;
    EXPECT(e3.eval({{"x", int64_t{3}}, {"y", int64_t{7}}}) == scalar{int64_t{21}});
    EXPECT(e4.eval({{"x", int64_t{3}}, {"y", int64_t{7}}}) == scalar{int64_t{21}});
}

TEST_CASE(canonical_interval_preserved)
{
    auto x    = var("x");
    auto y    = var("y");
    auto vars = std::unordered_map<std::string, interval>{{"x", interval{int64_t{1}, int64_t{3}}},
                                                          {"y", interval{int64_t{4}, int64_t{6}}}};
    EXPECT((x + y).eval_interval(vars) == (y + x).eval_interval(vars));
    EXPECT((x * y).eval_interval(vars) == (y * x).eval_interval(vars));
}

// ---- Algebraic normalization tests ----

TEST_CASE(norm_x_plus_x)
{
    auto x = var("x");
    EXPECT(x + x == lit(2) * x);
    EXPECT(x + x == x * lit(2));
}

TEST_CASE(norm_x_plus_2x)
{
    auto x = var("x");
    EXPECT(x + lit(2) * x == lit(3) * x);
    EXPECT(x + lit(2) * x == x + x + x);
}

TEST_CASE(norm_3x_minus_x)
{
    auto x = var("x");
    EXPECT(lit(3) * x - x == lit(2) * x);
}

TEST_CASE(norm_x_minus_x)
{
    auto x = var("x");
    EXPECT(x - x == lit(0));
}

TEST_CASE(norm_x_times_0)
{
    auto x = var("x");
    EXPECT(x * lit(0) == lit(0));
    EXPECT(lit(0) * x == lit(0));
}

TEST_CASE(norm_x_times_1)
{
    auto x = var("x");
    EXPECT(x * lit(1) == x);
    EXPECT(lit(1) * x == x);
}

TEST_CASE(norm_x_plus_0)
{
    auto x = var("x");
    EXPECT(x + lit(0) == x);
    EXPECT(lit(0) + x == x);
}

TEST_CASE(norm_distribute_simple)
{
    auto x = var("x");
    auto y = var("y");
    // 2*(x+y) == 2*x + 2*y
    EXPECT(lit(2) * (x + y) == lit(2) * x + lit(2) * y);
}

TEST_CASE(norm_foil)
{
    auto x = var("x");
    auto y = var("y");
    // (x+y)*(x+y) == x*x + 2*x*y + y*y
    EXPECT((x + y) * (x + y) == x * x + lit(2) * x * y + y * y);
}

TEST_CASE(norm_foil_eval)
{
    auto x    = var("x");
    auto y    = var("y");
    auto lhs  = (x + y) * (x + y);
    auto rhs  = x * x + lit(2) * x * y + y * y;
    auto vars = std::unordered_map<std::string, scalar>{{"x", int64_t{3}}, {"y", int64_t{4}}};
    EXPECT(lhs.eval(vars) == scalar{int64_t{49}});
    EXPECT(rhs.eval(vars) == scalar{int64_t{49}});
}

TEST_CASE(norm_difference_of_squares)
{
    auto x = var("x");
    auto y = var("y");
    // (x+y)*(x-y) == x*x - y*y
    EXPECT((x + y) * (x - y) == x * x - y * y);
}

TEST_CASE(norm_difference_of_squares_eval)
{
    auto x    = var("x");
    auto y    = var("y");
    auto lhs  = (x + y) * (x - y);
    auto rhs  = x * x - y * y;
    auto vars = std::unordered_map<std::string, scalar>{{"x", int64_t{5}}, {"y", int64_t{3}}};
    EXPECT(lhs.eval(vars) == scalar{int64_t{16}});
    EXPECT(rhs.eval(vars) == scalar{int64_t{16}});
}

TEST_CASE(norm_triple_product)
{
    auto x = var("x");
    // (x+1)*(x+1)*(x+1) expanded is x^3 + 3x^2 + 3x + 1
    auto cubed    = (x + lit(1)) * (x + lit(1)) * (x + lit(1));
    auto expanded = x * x * x + lit(3) * x * x + lit(3) * x + lit(1);
    EXPECT(cubed == expanded);
}

TEST_CASE(norm_triple_product_eval)
{
    auto x      = var("x");
    auto cubed  = (x + lit(1)) * (x + lit(1)) * (x + lit(1));
    auto result = cubed.eval({{"x", int64_t{2}}});
    EXPECT(result == scalar{int64_t{27}});
}

TEST_CASE(norm_collect_multi_var)
{
    auto x = var("x");
    auto y = var("y");
    // 2*x*y + 3*x*y == 5*x*y
    EXPECT(lit(2) * x * y + lit(3) * x * y == lit(5) * x * y);
}

TEST_CASE(norm_collect_mixed)
{
    auto x = var("x");
    auto y = var("y");
    // x + y + 2*x + 3*y == 3*x + 4*y
    EXPECT(x + y + lit(2) * x + lit(3) * y == lit(3) * x + lit(4) * y);
}

TEST_CASE(norm_nested_distribute)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    // a*(b+c) == a*b + a*c
    EXPECT(a * (b + c) == a * b + a * c);
}

TEST_CASE(norm_three_binomial)
{
    auto x = var("x");
    auto y = var("y");
    auto z = var("z");
    // (x+y)*(y+z) == x*y + x*z + y*y + y*z
    EXPECT((x + y) * (y + z) == x * y + x * z + y * y + y * z);
}

TEST_CASE(norm_subtract_expanded)
{
    auto x = var("x");
    auto y = var("y");
    // (x+y)*(x+y) - (x*x + y*y) == 2*x*y
    EXPECT((x + y) * (x + y) - (x * x + y * y) == lit(2) * x * y);
}

TEST_CASE(norm_negate_sum)
{
    auto x = var("x");
    auto y = var("y");
    // -(x + y) == -x + -y == -x - y
    EXPECT(-(x + y) == -x - y);
}

TEST_CASE(norm_double_negate)
{
    auto x = var("x");
    // -(-x) == x
    EXPECT(-(-x) == x);
}

TEST_CASE(norm_coefficient_fold)
{
    auto x = var("x");
    // 2 * 3 * x == 6 * x
    EXPECT(lit(2) * lit(3) * x == lit(6) * x);
}

TEST_CASE(norm_constant_add_in_sum)
{
    auto x = var("x");
    // (x + 3) + 5 == x + 8
    EXPECT(x + lit(3) + lit(5) == x + lit(8));
}

TEST_CASE(norm_zero_product_sum)
{
    auto x = var("x");
    auto y = var("y");
    // x*y - x*y == 0
    EXPECT(x * y - x * y == lit(0));
}

// ---- Division normalization tests ----

TEST_CASE(norm_div_identity)
{
    auto x = var("x");
    EXPECT(x / lit(1) == x);
}

TEST_CASE(norm_div_zero_numerator)
{
    auto x = var("x");
    EXPECT(lit(0) / x == lit(0));
    EXPECT(lit(0) / (x + lit(1)) == lit(0));
}

TEST_CASE(norm_div_self)
{
    auto x = var("x");
    EXPECT(x / x == lit(1));
    EXPECT((x + lit(1)) / (x + lit(1)) == lit(1));
}

TEST_CASE(norm_div_cancel_symbolic_factor)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(lit(2) * h / h == lit(2));
    EXPECT(h * w / h == w);
    EXPECT(h * w / w == h);
    EXPECT(lit(3) * h * w / h == lit(3) * w);
    EXPECT(lit(3) * h * w / (h * w) == lit(3));
}

TEST_CASE(norm_div_cancel_coefficient)
{
    auto n = var("n");
    EXPECT((lit(6) * n) / lit(3) == lit(2) * n);
    EXPECT((lit(6) * n) / lit(2) == lit(3) * n);
}

TEST_CASE(norm_div_cancel_mixed)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(h * lit(6) * w / (lit(3) * w) == lit(2) * h);
    EXPECT(h * h * w / (h * w) == h);
}

TEST_CASE(norm_div_cancel_partial)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(lit(5) * h * w / (lit(2) * h) == lit(5) * w / lit(2));
    EXPECT(h * h * w / (lit(2) * h) == h * w / lit(2));
}

TEST_CASE(norm_div_cancel_cross_factor)
{
    auto h = var("h");
    auto w = var("w");
    auto c = var("c");
    EXPECT(h * w / (h * c) == w / c);
    EXPECT(h * w / (h * h) == w / h);
}

TEST_CASE(norm_div_distribute_over_sum)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT((lit(2) * h + lit(4)) / lit(2) == h + lit(2));
    EXPECT((lit(6) * h + lit(3) * w + lit(9)) / lit(3) == lit(2) * h + w + lit(3));
    EXPECT((lit(4) * h + lit(2)) / lit(2) == lit(2) * h + lit(1));
}

TEST_CASE(norm_div_no_distribute_not_all_divisible)
{
    auto h = var("h");
    // (2*h + 3) / 2: 3 is not divisible by 2, so no distribution
    auto r = (lit(2) * h + lit(3)) / lit(2);
    EXPECT(r != h);
}

TEST_CASE(norm_div_constant_folding)
{
    EXPECT(lit(7) / lit(2) == lit(3));
    EXPECT(lit(6) / lit(3) == lit(2));
    EXPECT(lit(0) / lit(5) == lit(0));
}

// ---- Rewrite DSL tests ----

TEST_CASE(dsl_pvar_match)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    // sqrt(x) matches sqrt(_1)
    auto result = simplify(sqrt(x))(sqrt(_1) >> _1);
    EXPECT(result == x);
}

TEST_CASE(dsl_consistent_binding)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    auto y  = var("y");
    // _1 * _1 matches x*x but not x*y
    auto rule = _1 * _1 >> _1;
    EXPECT(simplify(x * x, {rule}) == x);
    EXPECT(simplify(x * y, {rule}) == x * y);
}

TEST_CASE(dsl_log_exp)
{
    auto _1     = pvar(1);
    auto x      = var("x");
    auto result = simplify(log(exp(x)))(log(exp(_1)) >> _1);
    EXPECT(result == x);
}

TEST_CASE(dsl_exp_log)
{
    auto _1     = pvar(1);
    auto x      = var("x");
    auto result = simplify(exp(log(x)))(exp(log(_1)) >> _1);
    EXPECT(result == x);
}

TEST_CASE(dsl_sqrt_product)
{
    auto _1     = pvar(1);
    auto _2     = pvar(2);
    auto a      = var("a");
    auto b      = var("b");
    auto result = simplify(sqrt(a * b))(sqrt(_1 * _2) >> sqrt(_1) * sqrt(_2));
    EXPECT(result == sqrt(a) * sqrt(b));
}

TEST_CASE(dsl_sqrt_division)
{
    auto _1     = pvar(1);
    auto _2     = pvar(2);
    auto a      = var("a");
    auto b      = var("b");
    auto result = simplify(sqrt(a / b))(sqrt(_1 / _2) >> sqrt(_1) / sqrt(_2));
    EXPECT(result == sqrt(a) / sqrt(b));
}

TEST_CASE(dsl_recursive)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    auto y  = var("y");
    // Rule applied to subexpressions: log(exp(x)) + log(exp(y))
    auto e      = log(exp(x)) + log(exp(y));
    auto result = simplify(e)(log(exp(_1)) >> _1);
    EXPECT(result == x + y);
}

TEST_CASE(dsl_multiple_rules)
{
    auto _1 = pvar(1);
    auto _2 = pvar(2);
    auto x  = var("x");
    auto y  = var("y");
    // Chain: pow(x,2) → x*x, then abs(x*x) → x*x (already positive)
    auto result =
        simplify(abs(pow(x, y)))(pow(_1, _2) >> _1 * _2, abs(_1 * _2) >> abs(_1) * abs(_2));
    EXPECT(result == abs(x) * abs(y));
}

TEST_CASE(dsl_trig_identity)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    // sin(x)^2 + cos(x)^2 == 1
    auto e      = sin(x) * sin(x) + cos(x) * cos(x);
    auto result = simplify(e)(sin(_1) * sin(_1) + cos(_1) * cos(_1) >> lit(1));
    EXPECT(result == lit(1));
}

TEST_CASE(dsl_no_match)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    // Rule doesn't match, expression unchanged
    auto result = simplify(sin(x))(log(exp(_1)) >> _1);
    EXPECT(result == sin(x));
}

TEST_CASE(dsl_literal_pattern)
{
    auto _1     = pvar(1);
    auto x      = var("x");
    auto result = simplify(pow(x, lit(2)))(pow(_1, lit(2)) >> _1 * _1);
    EXPECT(result == x * x);
}

TEST_CASE(dsl_eval_after_simplify)
{
    auto _1     = pvar(1);
    auto _2     = pvar(2);
    auto x      = var("x");
    auto y      = var("y");
    auto e      = sqrt(x * y);
    auto result = simplify(e)(sqrt(_1 * _2) >> sqrt(_1) * sqrt(_2));
    // sqrt(4) * sqrt(9) = 2 * 3 = 6
    EXPECT(result.eval({{"x", 4.0}, {"y", 9.0}}) == scalar{6.0});
}

TEST_CASE(dsl_chained_simplify)
{
    auto _1 = pvar(1);
    auto x  = var("x");
    // exp(log(exp(log(x)))) with repeated rule application
    auto e      = exp(log(exp(log(x))));
    auto result = simplify(e)(exp(log(_1)) >> _1, log(exp(_1)) >> _1);
    EXPECT(result == x);
}

TEST_CASE(dsl_nested_subexpr)
{
    auto _1 = pvar(1);
    auto _2 = pvar(2);
    auto a  = var("a");
    auto b  = var("b");
    auto c  = var("c");
    auto d  = var("d");
    // sqrt(a*b) + sqrt(c*d): rule applied to both subexprs
    auto e      = sqrt(a * b) + sqrt(c * d);
    auto result = simplify(e)(sqrt(_1 * _2) >> sqrt(_1) * sqrt(_2));
    EXPECT(result == sqrt(a) * sqrt(b) + sqrt(c) * sqrt(d));
}

// ---- Built-in rewrite rule tests ----

TEST_CASE(builtin_sqrt_product)
{
    auto a = var("a");
    auto b = var("b");
    // sqrt(a*b) automatically rewrites to sqrt(a)*sqrt(b)
    EXPECT(sqrt(a * b) == sqrt(a) * sqrt(b));
}

TEST_CASE(builtin_sqrt_division)
{
    auto a = var("a");
    auto b = var("b");
    // sqrt(a/b) automatically rewrites to sqrt(a)/sqrt(b)
    EXPECT(sqrt(a / b) == sqrt(a) / sqrt(b));
}

TEST_CASE(builtin_log_exp)
{
    auto x = var("x");
    // log(exp(x)) automatically simplifies to x
    EXPECT(log(exp(x)) == x);
}

TEST_CASE(builtin_exp_log)
{
    auto x = var("x");
    // exp(log(x)) automatically simplifies to x
    EXPECT(exp(log(x)) == x);
}

TEST_CASE(builtin_sqrt_product_eval)
{
    auto a = var("a");
    auto b = var("b");
    // sqrt(a*b) == sqrt(a)*sqrt(b), verify eval
    auto e = sqrt(a * b);
    EXPECT(e.eval({{"a", 4.0}, {"b", 9.0}}) == scalar{6.0});
}

TEST_CASE(builtin_log_exp_nested)
{
    auto x = var("x");
    auto y = var("y");
    // log(exp(x)) + log(exp(y)) automatically simplifies to x + y
    EXPECT(log(exp(x)) + log(exp(y)) == x + y);
}

TEST_CASE(builtin_raw_no_leak)
{
    auto x = var("x");
    // Ensure raw flag doesn't leak into normal expressions
    EXPECT(not x.is_raw());
    EXPECT(not(x + lit(1)).is_raw());
    EXPECT(not sqrt(x).is_raw());
}

TEST_CASE(builtin_pvar_is_raw)
{
    auto _1 = pvar(1);
    EXPECT(_1.is_raw());
    // Expressions built from pvars are raw
    EXPECT((_1 * pvar(2)).is_raw());
    EXPECT(sqrt(_1).is_raw());
}

// ---- Parse tests ----

TEST_CASE(parse_integer)
{
    auto e = parse("42");
    EXPECT(e == lit(42));
}

TEST_CASE(parse_double)
{
    auto e = parse("3.14");
    EXPECT(e == lit(3.14));
}

TEST_CASE(parse_variable)
{
    auto e = parse("x");
    EXPECT(e == var("x"));
}

TEST_CASE(parse_add)
{
    auto e = parse("x + y");
    EXPECT(e == var("x") + var("y"));
}

TEST_CASE(parse_sub)
{
    auto e = parse("x - y");
    EXPECT(e == var("x") - var("y"));
}

TEST_CASE(parse_mul)
{
    auto e = parse("x * y");
    EXPECT(e == var("x") * var("y"));
}

TEST_CASE(parse_div)
{
    auto e = parse("x / y");
    EXPECT(e == var("x") / var("y"));
}

TEST_CASE(parse_mod)
{
    auto e = parse("x % y");
    EXPECT(e == var("x") % var("y"));
}

TEST_CASE(parse_precedence)
{
    auto e = parse("x + y * z");
    EXPECT(e == var("x") + var("y") * var("z"));
}

TEST_CASE(parse_precedence_left)
{
    auto e = parse("x * y + z");
    EXPECT(e == var("x") * var("y") + var("z"));
}

TEST_CASE(parse_parens)
{
    auto e = parse("(x + y) * z");
    EXPECT(e == (var("x") + var("y")) * var("z"));
}

TEST_CASE(parse_nested_parens)
{
    auto e = parse("((x))");
    EXPECT(e == var("x"));
}

TEST_CASE(parse_unary_neg)
{
    auto e = parse("-x");
    EXPECT(e == -var("x"));
}

TEST_CASE(parse_neg_in_expr)
{
    auto e = parse("x + -y");
    EXPECT(e == var("x") + (-var("y")));
}

TEST_CASE(parse_function_sin)
{
    auto e = parse("sin(x)");
    EXPECT(e == sin(var("x")));
}

TEST_CASE(parse_function_cos)
{
    auto e = parse("cos(x)");
    EXPECT(e == cos(var("x")));
}

TEST_CASE(parse_function_sqrt)
{
    auto e = parse("sqrt(x)");
    EXPECT(e == sqrt(var("x")));
}

TEST_CASE(parse_function_exp)
{
    auto e = parse("exp(x)");
    EXPECT(e == exp(var("x")));
}

TEST_CASE(parse_function_log)
{
    auto e = parse("log(x)");
    EXPECT(e == log(var("x")));
}

TEST_CASE(parse_function_abs)
{
    auto e = parse("abs(x)");
    EXPECT(e == abs(var("x")));
}

TEST_CASE(parse_function_floor)
{
    auto e = parse("floor(x)");
    EXPECT(e == floor(var("x")));
}

TEST_CASE(parse_function_ceil)
{
    auto e = parse("ceil(x)");
    EXPECT(e == ceil(var("x")));
}

TEST_CASE(parse_function_tan)
{
    auto e = parse("tan(x)");
    EXPECT(e == tan(var("x")));
}

TEST_CASE(parse_function_pow)
{
    auto e = parse("pow(x, y)");
    EXPECT(e == pow(var("x"), var("y")));
}

TEST_CASE(parse_function_min)
{
    auto e = parse("min(x, y)");
    EXPECT(e == min(var("x"), var("y")));
}

TEST_CASE(parse_function_max)
{
    auto e = parse("max(x, y)");
    EXPECT(e == max(var("x"), var("y")));
}

TEST_CASE(parse_nested_functions)
{
    auto e = parse("sqrt(x * x + y * y)");
    EXPECT(e == sqrt(var("x") * var("x") + var("y") * var("y")));
}

TEST_CASE(parse_complex_expr)
{
    auto e = parse("sin(x) * cos(y) + 1");
    EXPECT(e == sin(var("x")) * cos(var("y")) + lit(1));
}

TEST_CASE(parse_whitespace_handling)
{
    auto e = parse("  x  +  y  ");
    EXPECT(e == var("x") + var("y"));
}

TEST_CASE(parse_no_whitespace)
{
    auto e = parse("x+y*z");
    EXPECT(e == var("x") + var("y") * var("z"));
}

TEST_CASE(parse_literal_arithmetic)
{
    auto e = parse("2 + 3");
    EXPECT(e == lit(5));
}

TEST_CASE(parse_roundtrip)
{
    auto x        = var("x");
    auto y        = var("y");
    auto original = sin(x) + cos(y) * lit(2);
    auto str      = to_string(original);
    auto parsed   = parse(str);
    EXPECT(parsed == original);
}

// ---- Serialization tests (to_value / from_value) ----

TEST_CASE(serialize_interval_int)
{
    interval i{int64_t{3}, int64_t{10}};
    auto v  = migraphx::to_value(i);
    auto i2 = migraphx::from_value<interval>(v);
    EXPECT(i == i2);
}

TEST_CASE(serialize_interval_double)
{
    interval i{1.5, 3.5};
    auto v  = migraphx::to_value(i);
    auto i2 = migraphx::from_value<interval>(v);
    EXPECT(i == i2);
}

TEST_CASE(serialize_interval_mixed)
{
    interval i{int64_t{0}, 5.5};
    auto v  = migraphx::to_value(i);
    auto i2 = migraphx::from_value<interval>(v);
    EXPECT(i == i2);
}

TEST_CASE(serialize_expr_literal_int)
{
    auto e  = lit(42);
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_literal_double)
{
    auto e  = lit(3.14);
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_variable)
{
    auto e  = var("x");
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_add)
{
    auto e  = var("x") + lit(1);
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_compound)
{
    auto e  = (var("x") + lit(1)) * var("y");
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_function)
{
    auto e  = sin(var("x"));
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_nested_function)
{
    auto e  = sqrt(var("x") + lit(1));
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_empty)
{
    expr e;
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e2.empty());
}

TEST_CASE(serialize_expr_eval_preserved)
{
    auto e  = var("x") * lit(2) + lit(3);
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e2.eval({{"x", int64_t{5}}}) == scalar{int64_t{13}});
}

TEST_CASE(serialize_expr_mod)
{
    auto e  = var("x") % lit(3);
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

TEST_CASE(serialize_expr_variable_constraint)
{
    auto e  = var("x", interval{int64_t{0}, int64_t{10}});
    auto v  = migraphx::to_value(e);
    auto e2 = migraphx::from_value<expr>(v);
    EXPECT(e == e2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
