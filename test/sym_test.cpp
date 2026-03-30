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
#include "test.hpp"

using se = migraphx::sym::expr;
using migraphx::sym::lit;
using migraphx::sym::parse;
using migraphx::sym::var;

// ===================================================================
// Tier 1: Expression construction and canonicalization
// ===================================================================

TEST_CASE(construct_integer)
{
    EXPECT(lit(0).to_string() == "0");
    EXPECT(lit(1).to_string() == "1");
    EXPECT(lit(42).to_string() == "42");
}

TEST_CASE(construct_symbol)
{
    EXPECT(var("h").to_string() == "h");
    EXPECT(var("batch_size").to_string() == "batch_size");
}

TEST_CASE(construct_empty)
{
    se e;
    EXPECT(e.empty());
    EXPECT(e.to_string().empty());
}

TEST_CASE(add_identity)
{
    auto h = var("h");
    EXPECT(h + 0 == h);
    EXPECT(0 + h == h);
}

TEST_CASE(add_commutativity)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(h + w == w + h);
}

TEST_CASE(add_like_term_folding)
{
    auto h = var("h");
    auto r = h + h;
    EXPECT(r == 2 * h);
}

TEST_CASE(add_constant_folding)
{
    EXPECT(lit(3) + lit(5) == lit(8));
    EXPECT(lit(0) + lit(0) == lit(0));
}

TEST_CASE(add_flattening)
{
    auto h = var("h");
    auto w = var("w");
    auto c = var("c");
    auto r = (h + w) + c;
    auto s = h + (w + c);
    auto t = (c + h) + w;
    EXPECT(r == s);
    EXPECT(s == t);
}

TEST_CASE(add_mixed)
{
    auto h = var("h");
    auto r = h + 3 + h + 2;
    EXPECT(r == 2 * h + 5);
    auto r2 = h + h;
    EXPECT(r2 + 5 == 2 * h + 5);
}

TEST_CASE(add_cancellation)
{
    auto h = var("h");
    EXPECT(h + (-1 * h) == lit(0));
}

TEST_CASE(sub_identity)
{
    auto h = var("h");
    EXPECT(h - 0 == h);
}

TEST_CASE(sub_self)
{
    auto h = var("h");
    EXPECT(h - h == lit(0));
}

TEST_CASE(sub_constant_folding) { EXPECT(lit(10) - lit(3) == lit(7)); }

TEST_CASE(sub_produces_negation)
{
    auto h = var("h");
    EXPECT(-1 * h == lit(0) - h);
    EXPECT(3 - h == lit(0) - h + 3);
}

TEST_CASE(neg_integer)
{
    auto r = lit(0) - 5;
    EXPECT(r == lit(-5));
}

TEST_CASE(neg_double_negation)
{
    auto h = var("h");
    auto r = 0 - (-1 * h);
    EXPECT(r == h);
}

TEST_CASE(mul_identity)
{
    auto h = var("h");
    EXPECT(h * 1 == h);
    EXPECT(1 * h == h);
}

TEST_CASE(mul_zero)
{
    auto h = var("h");
    EXPECT(h * 0 == lit(0));
    EXPECT(0 * h == lit(0));
}

TEST_CASE(mul_constant_folding) { EXPECT(lit(3) * lit(7) == lit(21)); }

TEST_CASE(mul_commutativity)
{
    auto a = var("a");
    auto b = var("b");
    EXPECT(b * a == a * b);
}

TEST_CASE(mul_coefficient_accumulation)
{
    auto h = var("h");
    auto r = 2 * h * 3;
    EXPECT(r == 6 * h);
}

TEST_CASE(mul_flattening)
{
    auto h = var("h");
    auto w = var("w");
    auto c = var("c");
    auto r = (h * w) * c;
    auto s = h * (w * c);
    auto t = (c * h) * w;
    EXPECT(r == s);
    EXPECT(s == t);
}

TEST_CASE(mul_distributive)
{
    auto h = var("h");
    auto r = 2 * (h + 1);
    EXPECT(r == 2 * h + 2);
}

TEST_CASE(fdiv_identity)
{
    auto h = var("h");
    EXPECT(h / 1 == h);
}

TEST_CASE(fdiv_constant_folding)
{
    EXPECT(lit(7) / lit(2) == lit(3));
    EXPECT(lit(6) / lit(3) == lit(2));
    EXPECT(lit(0) / lit(5) == lit(0));
}

TEST_CASE(fdiv_exact_coefficient_cancel)
{
    auto n = var("n");
    auto r = (6 * n) / 3;
    EXPECT(r == 2 * n);
}

TEST_CASE(fdiv_non_simplifiable)
{
    auto h = var("h");
    auto r = (h - 1) / 2;
    EXPECT(r == (h - 1) / 2);
}

TEST_CASE(fdiv_division_by_zero)
{
    EXPECT(test::throws([&] { var("h") / 0; }));
}

TEST_CASE(add_scaled_subtraction)
{
    auto h = var("h");
    EXPECT(2 * h - h == h);
    EXPECT(3 * h - 2 * h == h);
    EXPECT(h + h + h == 3 * h);
}

TEST_CASE(add_of_two_adds)
{
    auto h = var("h");
    auto r = (h + 1) + (h + 2);
    EXPECT(r == 2 * h + 3);
}

TEST_CASE(sub_strip_constant)
{
    auto h = var("h");
    EXPECT((h + 1) - h == lit(1));
}

TEST_CASE(sub_of_two_adds)
{
    auto h = var("h");
    auto r = (h + 1) - (h + 2);
    EXPECT(r == lit(-1));
}

TEST_CASE(mul_zero_propagation)
{
    auto h = var("h");
    EXPECT(50 * (h - h) == lit(0));
}

TEST_CASE(add_chain_constant_cancel)
{
    auto h = var("h");
    auto r = lit(2) - h - lit(2);
    EXPECT(r == -1 * h);
}

TEST_CASE(neg_of_sum_distributes)
{
    auto h = var("h");
    auto r = lit(-1) * (h + 1);
    EXPECT(r == -1 * h - 1);
}

TEST_CASE(neg_of_product_double)
{
    auto hw  = var("h") * var("w");
    auto neg = 0 - hw;
    EXPECT(neg == lit(0) - hw);
    auto dbl = 0 - neg;
    EXPECT(dbl == hw);
}

TEST_CASE(neg_of_neg_mul_canonicalizes)
{
    auto h   = var("h");
    auto neg = 0 - h;
    EXPECT(neg == lit(-1) * h);
    auto pos = 0 - neg;
    EXPECT(pos == h);
}

TEST_CASE(add_compound_product_like_terms)
{
    auto h  = var("h");
    auto w  = var("w");
    auto hw = h * w;
    auto wh = w * h;
    EXPECT(hw + hw == 2 * hw);
    EXPECT(hw + wh == 2 * hw);
    EXPECT(hw + 2 * hw == 3 * hw);
}

TEST_CASE(add_compound_product_cancellation)
{
    auto hw = var("h") * var("w");
    EXPECT(hw - hw == lit(0));
}

// X*Y and X cancel pairwise: (X*Y - X) - (X*Y - X) == 0
TEST_CASE(sub_compound_product_mixed)
{
    auto x  = var("x");
    auto y  = var("y");
    auto xy = x * y;
    auto r  = xy - x - xy + x;
    EXPECT(r == lit(0));
}

// Duplicate A*B terms fold even when separated by another term
TEST_CASE(add_multi_term_accumulation)
{
    auto a        = var("a");
    auto b        = var("b");
    auto c        = var("c");
    auto r        = a * b + c + a * b;
    auto expected = 2 * (a * b) + c;
    EXPECT(r == expected);
}

TEST_CASE(fdiv_negative_constant_folding)
{
    EXPECT(lit(-7) / lit(2) == lit(-7 / 2));
    EXPECT(lit(-6) / lit(3) == lit(-2));
    EXPECT(lit(7) / lit(-2) == lit(7 / -2));
}

TEST_CASE(fdiv_large_constants)
{
    EXPECT(lit(1000000) / lit(1000) == lit(1000));
    EXPECT(lit(999999) / lit(1000) == lit(999));
}

// ===================================================================
// Tier 2: Equality and hashing
// ===================================================================

TEST_CASE(eq_different_values)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(h + 1 != h + 2);
    EXPECT(h != w);
    EXPECT(lit(3) != lit(4));
}

TEST_CASE(eq_empty)
{
    EXPECT(se{} == se{});
    EXPECT(se{} != lit(0));
    EXPECT(lit(0) != se{});
}

// ===================================================================
// Tier 3: Evaluation and substitution
// ===================================================================

TEST_CASE(eval_simple)
{
    auto h = var("h");
    EXPECT(h.eval_dim({{h, 26}}) == 26);
    EXPECT(lit(42).eval_dim({}) == 42);
}

TEST_CASE(eval_arithmetic)
{
    auto h = var("h");
    EXPECT((h - 3).eval_dim({{h, 26}}) == 23);
    EXPECT((h + 5).eval_dim({{h, 10}}) == 15);
    EXPECT((2 * h).eval_dim({{h, 13}}) == 26);
}

TEST_CASE(eval_compound)
{
    auto h = var("h");
    auto e = (h - 3) / 2 + 1;
    EXPECT(e.eval_dim({{h, 26}}) == 12);
    EXPECT(e.eval_dim({{h, 27}}) == 13);
}

TEST_CASE(eval_multiple_symbols)
{
    auto n = var("n");
    auto h = var("h");
    auto e = n * h;
    EXPECT(e.eval_dim({{n, 4}, {h, 26}}) == 104);
}

TEST_CASE(eval_floor_division)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    EXPECT(e.eval_dim({{h, 7}}) == 3);
    EXPECT(e.eval_dim({{h, 8}}) == 3);
    EXPECT(e.eval_dim({{h, 9}}) == 4);
}

TEST_CASE(eval_unbound_throws)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(test::throws([&] { h.eval_dim({}); }));
    EXPECT(test::throws([&] { (h + w).eval_dim({{h, 1}}); }));
}

TEST_CASE(eval_division_by_zero_throws)
{
    auto h = var("h");
    auto d = var("d");
    EXPECT(test::throws([&] { (h / d).eval_dim({{h, 10}, {d, 0}}); }));
}

TEST_CASE(eval_integer_expr)
{
    EXPECT(lit(0).eval_dim({}) == 0);
    EXPECT(lit(100).eval_dim({}) == 100);
}

TEST_CASE(subs_partial)
{
    auto n = var("n");
    auto h = var("h");
    auto e = n * h + 1;
    auto r = e.subs({{n, lit(4)}});
    EXPECT(r == 4 * h + 1);
    EXPECT(r.eval_dim({{h, 10}}) == 41);
}

TEST_CASE(subs_full)
{
    auto h = var("h");
    auto e = h + 1;
    auto r = e.subs({{h, lit(5)}});
    EXPECT(r == lit(6));
}

TEST_CASE(subs_none)
{
    auto h = var("h");
    EXPECT(h.subs({}) == h);
}

TEST_CASE(subs_floor_div)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    auto r = e.subs({{h, lit(7)}});
    EXPECT(r == lit(3));
}

// eval() and subs()+eval() must agree on a compound expression
TEST_CASE(subs_eval_cross_validation)
{
    auto n                                       = var("n");
    auto h                                       = var("h");
    auto e                                       = (n * h - 3) / 2 + 1;
    std::unordered_map<se, std::size_t> eval_map = {{n, 4}, {h, 26}};
    std::unordered_map<se, se> subs_map          = {{n, lit(4)}, {h, lit(26)}};
    auto via_eval                                = e.eval_dim(eval_map);
    auto via_subs                                = e.subs(subs_map).eval_dim({});
    EXPECT(via_eval == via_subs);
}

TEST_CASE(subs_empty)
{
    se e;
    auto r = e.subs({{var("h"), lit(5)}});
    EXPECT(r.empty());
}

TEST_CASE(subs_creates_like_terms)
{
    auto h = var("h");
    auto w = var("w");
    auto e = h + w;
    auto r = e.subs({{w, lit(0)}});
    EXPECT(r == h);
}

TEST_CASE(subs_with_expression)
{
    auto h = var("h");
    auto w = var("w");
    auto e = 2 * h + 1;
    auto r = e.subs({{h, w + 3}});
    EXPECT(r == 2 * w + 7);
}

TEST_CASE(subs_symbol_for_symbol)
{
    auto h = var("h");
    auto w = var("w");
    auto e = h * h + 1;
    auto r = e.subs({{h, w}});
    EXPECT(r == w * w + 1);
}

TEST_CASE(subs_compound_expression)
{
    auto n = var("n");
    auto h = var("h");
    auto w = var("w");
    auto e = (n * h + w - 3) / 2;
    auto r = e.subs({{h, 2 * w + 1}, {n, w - 1}});
    // N*H => (W-1)*(2*W+1) = 2*W^2 - W - 1
    // N*H + W - 3 => 2*W^2 - 2*W - 4 + W = 2*W^2 - 2
    // Verify by evaluating with W=5: (W-1)*(2*W+1) + W - 3 = 4*11 + 5 - 3 = 46, 46/2 = 23
    EXPECT(r.eval_dim({{w, 5}}) == 23);
    // Also verify the original expression with direct values agrees
    EXPECT(e.eval_dim({{n, 4}, {h, 11}, {w, 5}}) == 23);
}

TEST_CASE(eval_compound_product)
{
    auto h = var("h");
    auto w = var("w");
    auto e = h * w + 1;
    EXPECT(e.eval_dim({{h, 3}, {w, 4}}) == 13);
}

TEST_CASE(eval_negative_intermediate)
{
    auto h = var("h");
    auto e = (h - 10) * 2 + 20;
    EXPECT(e.eval_dim({{h, 3}}) == 6);
}

// ===================================================================
// Tier 4: Printing and parsing
// ===================================================================

TEST_CASE(print_atoms)
{
    EXPECT(lit(42).to_string() == "42");
    EXPECT(var("h").to_string() == "h");
    EXPECT(lit(0).to_string() == "0");
    EXPECT(lit(-3).to_string() == "-3");
}

TEST_CASE(print_add)
{
    auto h = var("h");
    EXPECT((h + 1).to_string() == "h + 1");
    EXPECT((h - 3).to_string() == "h - 3");
}

TEST_CASE(print_mul)
{
    EXPECT((2 * var("h")).to_string() == "2*h");
    auto r = var("a") * var("b");
    EXPECT(r.to_string() == "a*b");
}

TEST_CASE(print_fdiv_parens)
{
    auto r = (var("h") - 1) / 2;
    EXPECT(r.to_string() == "(h - 1)/2");
}

TEST_CASE(print_compound)
{
    auto r = (var("h") - 3) / 2 + 1;
    EXPECT(r.to_string() == "(h - 3)/2 + 1");
}

TEST_CASE(parse_atoms)
{
    EXPECT(parse("42") == lit(42));
    EXPECT(parse("h") == var("h"));
}

TEST_CASE(parse_arithmetic)
{
    auto h = var("h");
    auto r = parse("h + 1");
    EXPECT(r == h + 1);

    auto r2 = parse("h - 3");
    EXPECT(r2 == h - 3);

    auto r3 = parse("2*h");
    EXPECT(r3 == 2 * h);
}

TEST_CASE(parse_precedence)
{
    auto r = parse("h + 1 * 2");
    EXPECT(r == var("h") + 2);
}

TEST_CASE(parse_parentheses)
{
    auto r = parse("(h + 1) * 2");
    EXPECT(r == 2 * (var("h") + 1));
}

TEST_CASE(parse_division)
{
    auto r = parse("(h - 1)/2");
    EXPECT(r == (var("h") - 1) / 2);
}

TEST_CASE(parse_unary_minus)
{
    auto h = var("h");
    EXPECT(parse("-h") == -1 * h);
    EXPECT(parse("-h").to_string() == "-h");
    EXPECT(parse("-(h + 1)") == -1 * h - 1);
}

TEST_CASE(parse_floor_backward_compat)
{
    auto a = parse("floor((h-1)/2)");
    auto b = parse("(h-1)/2");
    EXPECT(a == b);

    auto c = parse("floor((h-1)/2) + 1");
    auto d = (var("h") - 1) / 2 + 1;
    EXPECT(c == d);
}

TEST_CASE(parse_whitespace_tolerance)
{
    EXPECT(parse("  h  +  1  ") == parse("h + 1"));
    EXPECT(parse("h+1") == parse("h + 1"));
}

TEST_CASE(parse_power_operator)
{
    auto h = var("h");
    EXPECT(parse("h**2") == h * h);
    EXPECT(parse("h**3") == h * h * h);
    EXPECT(parse("h**1") == h);
    EXPECT(parse("h**0") == lit(1));
    EXPECT(parse("2*h**2 + 1") == 2 * h * h + 1);
    EXPECT(parse("(2*h)**3 + 5") == 8 * h * h * h + 5);
}

TEST_CASE(print_negative_mul_coefficient)
{
    auto r = 0 - 3 * var("h");
    EXPECT(r.to_string() == "-3*h");
}

TEST_CASE(print_multi_symbol_product)
{
    auto r = var("h") * var("w");
    auto s = r.to_string();
    EXPECT(s == "h*w" or s == "w*h");
    EXPECT(parse("h*w") == parse("w*h"));
}

TEST_CASE(print_compound_expression)
{
    auto r = 2 * (var("h") * var("w")) + var("c") - 1;
    auto s = r.to_string();
    EXPECT(parse(s) == r);
}

TEST_CASE(parse_compound_mul)
{
    auto r = parse("2*h*w");
    EXPECT(r == 2 * var("h") * var("w"));
}

TEST_CASE(print_parse_round_trip)
{
    auto h                = var("h");
    auto n                = var("n");
    auto c                = var("c");
    auto w                = var("w");
    std::vector<se> exprs = {
        h,
        h + 1,
        2 * h - 3,
        (h - 3) / 2 + 1,
        n * c * h * w,
        (h - 1) / 2,
        h * h,
        h * h * h,
    };
    for(const auto& e : exprs)
    {
        auto s        = e.to_string();
        auto reparsed = parse(s);
        EXPECT(reparsed == e);
    }
}

// ===================================================================
// Tier 6: Edge cases and robustness
// ===================================================================

// 5 levels of (e-1)/2: simulates repeated pooling/conv stride reduction
TEST_CASE(edge_deeply_nested)
{
    auto h = var("h");
    se e   = h;
    for(int i = 0; i < 5; ++i)
        e = (e - 1) / 2;
    EXPECT(e.eval_dim({{h, 255}}) == 7);
}

TEST_CASE(edge_many_symbols)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    auto e = var("e");
    auto r = a + b + c + d + e;
    EXPECT(r.eval_dim({{a, 1}, {b, 2}, {c, 3}, {d, 4}, {e, 5}}) == 15);
}

TEST_CASE(edge_neg_one_coefficient)
{
    auto h = var("h");
    EXPECT(-1 * h == lit(0) - h);
    EXPECT(-1 * h + h == lit(0));
}

TEST_CASE(edge_empty_operations)
{
    se empty;
    EXPECT((empty + empty).empty());
    EXPECT((empty - empty).empty());
    EXPECT((empty * empty).empty());
    EXPECT((empty / empty).empty());
}

TEST_CASE(edge_empty_with_nonempty)
{
    se empty;
    auto h  = var("h");
    auto r1 = h + empty;
    EXPECT(not r1.empty());

    auto r2 = empty + h;
    EXPECT(not r2.empty());
}

TEST_CASE(edge_large_coefficients)
{
    auto h = var("h");
    auto r = 1000000 * h;
    EXPECT(r.eval_dim({{h, 1000000}}) == 1000000000000ULL);
}

// Incrementally adding H ten times must fold to 11*H
TEST_CASE(edge_chained_operations)
{
    auto h = var("h");
    auto e = h;
    for(int i = 0; i < 10; ++i)
        e = e + h;
    EXPECT(e == 11 * h);
}

TEST_CASE(edge_repeated_parse)
{
    auto h = var("h");
    for(int i = 0; i < 10; ++i)
    {
        auto r = parse("(h - 3)/2 + 1");
        EXPECT(r == (h - 3) / 2 + 1);
    }
}

// ===================================================================
// Serialization round-trip
// ===================================================================

static se round_trip(const se& e)
{
    auto v = migraphx::to_value(e);
    return migraphx::from_value<se>(v);
}

TEST_CASE(serialize_empty)
{
    se e;
    EXPECT(round_trip(e).empty());
}

TEST_CASE(serialize_integer)
{
    EXPECT(round_trip(lit(0)) == lit(0));
    EXPECT(round_trip(lit(42)) == lit(42));
}

TEST_CASE(serialize_symbol)
{
    auto h = var("h");
    EXPECT(round_trip(h) == h);
}

TEST_CASE(serialize_add)
{
    auto h = var("h");
    auto e = 2 * h + 3;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_mul)
{
    auto h = var("h");
    auto w = var("w");
    auto e = h * w;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_fdiv)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_compound)
{
    auto n = var("n");
    auto h = var("h");
    auto w = var("w");
    auto e = (n * h * w + 3) / 2 - 1;
    EXPECT(round_trip(e) == e);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
