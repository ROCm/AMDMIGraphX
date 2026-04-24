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

using se       = migraphx::sym::expr;
using interval = migraphx::sym::interval;
using migraphx::sym::lit;
using migraphx::sym::parse;

// Local wrappers so sym-library arithmetic/canonicalization tests don't have
// to spell out bounds they don't care about
static se var(const std::string& name) { return migraphx::sym::var(name, {1, 1}); }
static se var(const std::string& name, interval bounds, std::set<int64_t> optimals = {})
{
    return migraphx::sym::var(name, bounds, optimals);
}

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

TEST_CASE(construct_empty_var_name_throws)
{
    EXPECT(test::throws([&] { var(""); }));
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

TEST_CASE(tdiv_identity)
{
    auto h = var("h");
    EXPECT(h / 1 == h);
}

TEST_CASE(tdiv_constant_folding)
{
    EXPECT(lit(7) / lit(2) == lit(3));
    EXPECT(lit(6) / lit(3) == lit(2));
    EXPECT(lit(0) / lit(5) == lit(0));
}

TEST_CASE(tdiv_exact_coefficient_cancel)
{
    auto n = var("n");
    auto r = (6 * n) / 3;
    EXPECT(r == 2 * n);
}

TEST_CASE(tdiv_non_simplifiable)
{
    auto h = var("h");
    auto r = (h - 1) / 2;
    EXPECT(r == (h - 1) / 2);
}

TEST_CASE(tdiv_division_by_zero)
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

TEST_CASE(tdiv_negative_constant_folding)
{
    EXPECT(lit(-7) / lit(2) == lit(-7 / 2));
    EXPECT(lit(-6) / lit(3) == lit(-2));
    EXPECT(lit(7) / lit(-2) == lit(7 / -2));
}

TEST_CASE(tdiv_large_constants)
{
    EXPECT(lit(1000000) / lit(1000) == lit(1000));
    EXPECT(lit(999999) / lit(1000) == lit(999));
}

TEST_CASE(tdiv_zero_numerator)
{
    auto h = var("h");
    EXPECT(lit(0) / h == lit(0));
    EXPECT(lit(0) / (h + 1) == lit(0));
}

TEST_CASE(tdiv_self)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(h / h == lit(1));
    EXPECT((h + 1) / (h + 1) == lit(1));
    EXPECT((h * w) / (h * w) == lit(1));
}

TEST_CASE(tdiv_cancel_symbolic_factor)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(2 * h / h == lit(2));
    EXPECT(h * w / h == w);
    EXPECT(h * w / w == h);
    EXPECT(3 * h * w / h == 3 * w);
    EXPECT(3 * h * w / (h * w) == lit(3));
    EXPECT(h * 6 * w / (3 * w) == 2 * h);
    EXPECT(h * h * w / (h * w) == h);
}

TEST_CASE(tdiv_cancel_partial)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(5 * h * w / (2 * h) == 5 * w / lit(2));
    EXPECT(h * h * w / (2 * h) == h * w / lit(2));
}

TEST_CASE(tdiv_cancel_cross_factor)
{
    auto h = var("h");
    auto w = var("w");
    auto c = var("c");
    EXPECT(h * w / (h * c) == w / c);
    EXPECT(h * w / (h * h) == w / h);
}

TEST_CASE(tdiv_distribute_over_sum)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT((2 * h + 4) / 2 == h + 2);
    EXPECT((6 * h + 3 * w + 9) / 3 == 2 * h + w + 3);
    EXPECT((4 * h + 2) / 2 == 2 * h + 1);
    EXPECT((2 * h + 3) / 2 != h);
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

TEST_CASE(hash_consistency)
{
    auto h = var("h");
    auto w = var("w");
    auto n = var("n");

    auto check = [](const se& a, const se& b) {
        EXPECT(a == b);
        EXPECT(a.hash() == b.hash());
    };

    check(h + w, w + h);
    check(h * w, w * h);
    check(2 * h + 3, 3 + 2 * h);
    check(h * w * n, n * h * w);
    check((h + 1) * 3, 3 * (h + 1));
    check((h - 1) / 2, (h - 1) / 2);
    check(h + 0, h);
    check(h * 1, h);
    check(lit(5), lit(5));
}

// ===================================================================
// Tier 3: Evaluation and substitution
// ===================================================================

TEST_CASE(eval_simple)
{
    auto h = var("h");
    EXPECT(h.eval_uint({{h, 26}}) == 26);
    EXPECT(lit(42).eval_uint({}) == 42);
}

TEST_CASE(eval_arithmetic)
{
    auto h = var("h");
    EXPECT((h - 3).eval_uint({{h, 26}}) == 23);
    EXPECT((h + 5).eval_uint({{h, 10}}) == 15);
    EXPECT((2 * h).eval_uint({{h, 13}}) == 26);
}

TEST_CASE(eval_compound)
{
    auto h = var("h");
    auto e = (h - 3) / 2 + 1;
    EXPECT(e.eval_uint({{h, 26}}) == 12);
    EXPECT(e.eval_uint({{h, 27}}) == 13);
}

TEST_CASE(eval_multiple_symbols)
{
    auto n = var("n");
    auto h = var("h");
    auto e = n * h;
    EXPECT(e.eval_uint({{n, 4}, {h, 26}}) == 104);
}

TEST_CASE(eval_trunc_division)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    EXPECT(e.eval_uint({{h, 7}}) == 3);
    EXPECT(e.eval_uint({{h, 8}}) == 3);
    EXPECT(e.eval_uint({{h, 9}}) == 4);
}

TEST_CASE(eval_unbound_throws)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(test::throws([&] { h.eval_uint({}); }));
    EXPECT(test::throws([&] { (h + w).eval_uint({{h, 1}}); }));
}

TEST_CASE(eval_division_by_zero_throws)
{
    auto h = var("h");
    auto d = var("d");
    EXPECT(test::throws([&] { (h / d).eval_uint({{h, 10}, {d, 0}}); }));
}

TEST_CASE(eval_integer_expr)
{
    EXPECT(lit(0).eval_uint({}) == 0);
    EXPECT(lit(100).eval_uint({}) == 100);
}

TEST_CASE(eval_non_symbol_key_throws)
{
    auto h = var("h");
    EXPECT(test::throws([&] { h.eval_uint({{lit(5), 10}}); }));
    EXPECT(test::throws([&] { h.eval_uint({{h + 1, 10}}); }));
}

TEST_CASE(subs_non_symbol_key_throws)
{
    auto h = var("h");
    EXPECT(test::throws([&] { h.subs({{h + 1, lit(5)}}); }));
    EXPECT(test::throws([&] { h.subs({{lit(3), lit(5)}}); }));
}

TEST_CASE(subs_partial)
{
    auto n = var("n");
    auto h = var("h");
    auto e = n * h + 1;
    auto r = e.subs({{n, lit(4)}});
    EXPECT(r == 4 * h + 1);
    EXPECT(r.eval_uint({{h, 10}}) == 41);
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

TEST_CASE(subs_trunc_div)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    auto r = e.subs({{h, lit(7)}});
    EXPECT(r == lit(3));
}

TEST_CASE(subs_division_by_zero)
{
    auto h = var("h");
    auto d = var("d");
    auto e = h / d;
    EXPECT(test::throws([&] { e.subs({{d, lit(0)}}); }));
}

// eval() and subs()+eval() must agree on a compound expression
TEST_CASE(subs_eval_cross_validation)
{
    auto n                                       = var("n");
    auto h                                       = var("h");
    auto e                                       = (n * h - 3) / 2 + 1;
    std::unordered_map<se, std::size_t> eval_map = {{n, 4}, {h, 26}};
    std::unordered_map<se, se> subs_map          = {{n, lit(4)}, {h, lit(26)}};
    auto via_eval                                = e.eval_uint(eval_map);
    auto via_subs                                = e.subs(subs_map).eval_uint({});
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
    EXPECT(r.eval_uint({{w, 5}}) == 23);
    // Also verify the original expression with direct values agrees
    EXPECT(e.eval_uint({{n, 4}, {h, 11}, {w, 5}}) == 23);
}

TEST_CASE(eval_compound_product)
{
    auto h = var("h");
    auto w = var("w");
    auto e = h * w + 1;
    EXPECT(e.eval_uint({{h, 3}, {w, 4}}) == 13);
}

TEST_CASE(eval_negative_intermediate)
{
    auto h = var("h");
    auto e = (h - 10) * 2 + 20;
    EXPECT(e.eval_uint({{h, 3}}) == 6);
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

TEST_CASE(print_tdiv_parens)
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

TEST_CASE(parse_whitespace_tolerance)
{
    EXPECT(parse("  h  +  1  ") == parse("h + 1"));
    EXPECT(parse("h+1") == parse("h + 1"));
}

TEST_CASE(parse_empty_string)
{
    EXPECT(parse("").empty());
    EXPECT(parse("  ").empty());
    EXPECT(parse("\t\n").empty());
}

TEST_CASE(parse_error_unexpected_char)
{
    EXPECT(test::throws([&] { parse(")"); }));
    EXPECT(test::throws([&] { parse("@"); }));
}

TEST_CASE(parse_error_trailing_chars)
{
    EXPECT(test::throws([&] { parse("1 2"); }));
    EXPECT(test::throws([&] { parse("h w"); }));
}

TEST_CASE(parse_error_unexpected_end)
{
    EXPECT(test::throws([&] { parse("h +"); }));
    EXPECT(test::throws([&] { parse("h *"); }));
    EXPECT(test::throws([&] { parse("-"); }));
}

TEST_CASE(parse_double_unary_minus)
{
    auto h = var("h");
    EXPECT(parse("--h") == h);
    EXPECT(parse("--5") == lit(5));
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
    EXPECT(e.eval_uint({{h, 255}}) == 7);
}

TEST_CASE(edge_many_symbols)
{
    auto a = var("a");
    auto b = var("b");
    auto c = var("c");
    auto d = var("d");
    auto e = var("e");
    auto r = a + b + c + d + e;
    EXPECT(r.eval_uint({{a, 1}, {b, 2}, {c, 3}, {d, 4}, {e, 5}}) == 15);
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
    auto h = var("h");
    EXPECT((h + empty).empty());
    EXPECT((empty + h).empty());
    EXPECT((h - empty).empty());
    EXPECT((empty * h).empty());
    EXPECT((h / empty).empty());
}

TEST_CASE(edge_large_coefficients)
{
    auto h = var("h");
    auto r = 1000000 * h;
    EXPECT(r.eval_uint({{h, 1000000}}) == 1000000000000ULL);
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

TEST_CASE(serialize_tdiv)
{
    auto h = var("h");
    auto e = (h - 1) / 2;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_negative_integer)
{
    EXPECT(round_trip(lit(-5)) == lit(-5));
    EXPECT(round_trip(lit(-1)) == lit(-1));
}

TEST_CASE(serialize_power)
{
    auto h = var("h");
    EXPECT(round_trip(h * h) == h * h);
    EXPECT(round_trip(h * h * h) == h * h * h);
}

TEST_CASE(serialize_negative_coefficient)
{
    auto h = var("h");
    EXPECT(round_trip(0 - 3 * h) == 0 - 3 * h);
    EXPECT(round_trip(0 - h) == 0 - h);
}

TEST_CASE(serialize_tdiv_symbolic_denominator)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(round_trip(h / w) == h / w);
    EXPECT(round_trip((h + 1) / w) == (h + 1) / w);
}

TEST_CASE(serialize_compound)
{
    auto n = var("n");
    auto h = var("h");
    auto w = var("w");
    auto e = (n * h * w + 3) / 2 - 1;
    EXPECT(round_trip(e) == e);
}

// -------------------------------------------------------------------
// Bounded vars: constructor / eq / hash
// -------------------------------------------------------------------

TEST_CASE(construct_var_min_greater_than_max_throws)
{
    EXPECT(test::throws([&] { var("n", {10, 5}); }));
}

TEST_CASE(construct_var_min_less_than_one_throws)
{
    EXPECT(test::throws([&] { var("n", {0, 5}); }));
    EXPECT(test::throws([&] { var("n", {-1, 5}); }));
}

TEST_CASE(eq_same_name_different_intervals)
{
    auto h1 = var("h", {1, 128});
    auto h2 = var("h", {1, 256});
    auto h3 = var("h", {2, 128});
    auto h4 = var("h", {1, 128});
    EXPECT(h1 != h2);
    EXPECT(h1 != h3);
    EXPECT(h1 == h4);
}

TEST_CASE(hash_same_name_different_intervals)
{
    auto h1 = var("h", {1, 128});
    auto h2 = var("h", {1, 256});
    auto h3 = var("h", {1, 128});
    EXPECT(h1.hash() != h2.hash());
    EXPECT(h1.hash() == h3.hash());
}

// -------------------------------------------------------------------
// Bounds: eval_interval()
// -------------------------------------------------------------------

TEST_CASE(eval_interval_single_var)
{
    auto n = var("n", {2, 16});
    EXPECT(n.eval_interval() == interval{2, 16});
}

TEST_CASE(eval_interval_literal) { EXPECT(lit(42).eval_interval() == interval{42, 42}); }

TEST_CASE(eval_interval_compound)
{
    auto n = var("n", {1, 8});
    auto c = var("c", {1, 16});
    auto e = n * c * 4;
    EXPECT(e.eval_interval() == interval{4, 512});
}

TEST_CASE(eval_interval_stride_diff)
{
    auto n    = var("n", {1, 8});
    auto c    = var("c", {1, 16});
    auto diff = n * c - n;
    EXPECT(diff.eval_interval() == interval{0, 120});
}

TEST_CASE(eval_interval_division)
{
    auto n = var("n", {2, 10});
    auto d = var("d", {1, 5});
    auto e = n / d;
    EXPECT(e.eval_interval() == interval{0, 10});
}

TEST_CASE(eval_interval_div_literal_denom)
{
    auto n = var("n", {4, 16});
    auto e = n / lit(4);
    EXPECT(e.eval_interval() == interval{1, 4});
}

TEST_CASE(eval_interval_subtraction_independent)
{
    auto a = var("a", {1, 10});
    auto b = var("b", {1, 5});
    auto e = a - b;
    EXPECT(e.eval_interval() == interval{-4, 9});
}

TEST_CASE(eval_interval_empty_throws)
{
    se empty;
    EXPECT(test::throws([&] { (void)empty.eval_interval(); }));
}

TEST_CASE(eval_interval_uint)
{
    auto n = var("n", {2, 16});
    auto e = 3 * n + 1;
    EXPECT(e.eval_interval() == interval{7, 49});
}

// -------------------------------------------------------------------
// Comparison operators
// -------------------------------------------------------------------

TEST_CASE(cmp_lit_constants)
{
    EXPECT(lit(1) < lit(2));
    EXPECT(not(lit(2) < lit(1)));
    EXPECT(not(lit(3) < lit(3)));
    EXPECT(lit(2) > lit(1));
    EXPECT(lit(3) <= lit(3));
    EXPECT(lit(3) >= lit(3));
    EXPECT(lit(1) <= lit(2));
    EXPECT(lit(2) >= lit(1));
}

TEST_CASE(cmp_equal_expr_not_less)
{
    auto n = var("n");
    EXPECT(not(n < n));
    EXPECT(not(n > n));
    EXPECT(n <= n);
    EXPECT(n >= n);
}

TEST_CASE(cmp_empty_not_less)
{
    se a;
    se b;
    EXPECT(not(a < b));
}

TEST_CASE(cmp_empty_with_nonempty_throws)
{
    EXPECT(test::throws([&]() -> bool { return se{} < var("n"); }));
    EXPECT(test::throws([&]() -> bool { return var("n") < se{}; }));
}

TEST_CASE(cmp_stride_ordering_4d)
{
    auto c  = var("c", {1, 512});
    auto h  = var("h", {1, 256});
    auto w  = var("w", {1, 256});
    auto s0 = c * h * w;
    auto s1 = h * w;
    auto s2 = w;
    auto s3 = lit(1);
    EXPECT(s1 <= s0);
    EXPECT(s2 <= s1);
    EXPECT(s3 <= s2);
    EXPECT(s3 <= s0);
}

TEST_CASE(cmp_scaled_symbol)
{
    auto n = var("n");
    EXPECT(n < 2 * n);
    EXPECT(n < 3 * n);
    EXPECT(not(2 * n < n));
}

TEST_CASE(cmp_product_explicit_bounds)
{
    auto k = var("k", {1, 8});
    auto m = var("m", {2, 4});
    EXPECT(k < m * k);
}

TEST_CASE(cmp_conv_output_smaller_than_input)
{
    auto h   = var("h", {3, 256});
    auto out = (h - 3) / 2 + 1;
    EXPECT(out < h);
    EXPECT(not(h < out));
}

TEST_CASE(cmp_repeated_pooling)
{
    auto h    = var("h", {7, 256});
    auto out1 = (h - 3) / 2 + 1;
    auto out2 = (out1 - 3) / 2 + 1;
    EXPECT(out1 < h);
    EXPECT(out2 < out1);
    EXPECT(out2 < h);
}

TEST_CASE(cmp_strides_after_conv)
{
    auto h     = var("h", {7, 128});
    auto w     = var("w", {2, 128});
    auto new_h = (h - 3) / 2 + 1;
    auto s0    = new_h * w;
    auto s1    = w;
    auto s2    = lit(1);
    EXPECT(s1 < s0);
    EXPECT(s2 < s1);
}

TEST_CASE(cmp_broadcast_stride_zero)
{
    auto w = var("w");
    EXPECT(lit(0) < w);
    EXPECT(not(w < lit(0)));
}

TEST_CASE(cmp_offset_expressions)
{
    auto h = var("h", {2, 256});
    EXPECT(h - 1 < h);
    EXPECT(h < h + 1);
    EXPECT(not(h + 1 < h));
}

TEST_CASE(cmp_undetermined_throws)
{
    auto n = var("n", {2, 10});
    EXPECT(test::throws([&]() -> bool { return n < lit(5); }));
}

TEST_CASE(cmp_element_count_slice)
{
    auto n = var("n", {1, 32});
    auto c = var("c", {1, 512});
    auto h = var("h", {1, 256});
    auto w = var("w", {2, 256});
    EXPECT(n * c * h < n * c * h * w);
}

TEST_CASE(cmp_deep_pooling_chain)
{
    auto h   = var("h", {31, 512});
    se stage = h;
    se prev;
    for(int i = 0; i < 5; ++i)
    {
        prev  = stage;
        stage = (stage - 1) / 2;
    }
    EXPECT(stage < prev);
    EXPECT(stage < h);
}

TEST_CASE(cmp_commuted_product)
{
    auto a = var("a");
    auto b = var("b");
    EXPECT(not(a * b < b * a));
    EXPECT(a * b <= b * a);
    EXPECT(a * b >= b * a);
}

TEST_CASE(cmp_negative_literals)
{
    EXPECT(lit(-5) < lit(-1));
    EXPECT(lit(-1) < lit(0));
    EXPECT(lit(-10) < lit(10));
    EXPECT(not(lit(0) < lit(-1)));
}

TEST_CASE(cmp_symmetry_lt_gt)
{
    auto h   = var("h", {3, 256});
    auto out = (h - 3) / 2 + 1;
    EXPECT(out < h);
    EXPECT(h > out);
    EXPECT(not(h < out));
    EXPECT(not(out > h));
}

TEST_CASE(cmp_transitivity_strides)
{
    auto c  = var("c", {2, 512});
    auto h  = var("h", {2, 256});
    auto w  = var("w", {2, 256});
    auto s0 = c * h * w;
    auto s1 = h * w;
    auto s2 = w;
    auto s3 = lit(1);
    EXPECT(s1 < s0);
    EXPECT(s2 < s1);
    EXPECT(s3 < s2);
    EXPECT(s3 < s0);
    EXPECT(s2 < s0);
    EXPECT(s3 < s1);
}

TEST_CASE(cmp_division_ordering)
{
    auto h     = var("h", {5, 256});
    auto pool2 = (h - 1) / 2;
    auto pool4 = (h - 1) / 4;
    EXPECT(pool4 < pool2);
    EXPECT(pool2 < h);
    EXPECT(pool4 < h);
}

TEST_CASE(cmp_sum_less_than_product)
{
    auto n = var("n", {2, 32});
    auto c = var("c", {3, 512});
    EXPECT(n + c < n * c);
}

TEST_CASE(cmp_algebraically_equal_expressions)
{
    auto h = var("h");
    auto a = h + h;
    auto b = 2 * h;
    EXPECT(a == b);
    EXPECT(not(a < b));
    EXPECT(not(b < a));
    EXPECT(a <= b);
    EXPECT(a >= b);
}

TEST_CASE(cmp_zero_stride_less_than_symbolic_stride)
{
    auto h = var("h");
    auto w = var("w");
    EXPECT(lit(0) < h);
    EXPECT(lit(0) < h * w);
    EXPECT(lit(0) < h + w);
}

// -------------------------------------------------------------------
// Optimals: eval_optimals()
// -------------------------------------------------------------------

TEST_CASE(eval_optimals_single_var)
{
    auto n = var("n", {1, 8}, {2, 4});
    EXPECT(n.eval_optimals() == std::set<std::size_t>{2, 4});
}

TEST_CASE(eval_optimals_compound_expr)
{
    auto n = var("n", {1, 8}, {2, 4});
    auto e = 2 * n + 1;
    EXPECT(e.eval_optimals() == std::set<std::size_t>{5, 9});
}

TEST_CASE(eval_optimals_multi_var)
{
    auto n = var("n", {1, 8}, {2, 4});
    auto m = var("m", {1, 8}, {3, 6});
    auto e = n + m;
    EXPECT(e.eval_optimals() == std::set<std::size_t>{5, 7, 8, 10});
}

TEST_CASE(eval_optimals_negative_throws)
{
    auto n = var("n", {1, 4}, {2});
    auto m = var("m", {1, 8}, {5});
    auto e = n - m;
    EXPECT(test::throws([&] { (void)e.eval_optimals(); }));
}

TEST_CASE(eval_optimals_no_optimals)
{
    auto n = var("n", {1, 8});
    EXPECT(n.eval_optimals().empty());
}

TEST_CASE(eval_optimals_empty_expr)
{
    se e;
    EXPECT(e.eval_optimals().empty());
}

// -------------------------------------------------------------------
// Serialization: bounded vars
// -------------------------------------------------------------------

TEST_CASE(serialize_bounded_var)
{
    auto h = var("h", {1, 128});
    auto r = round_trip(h);
    EXPECT(r == h);
    EXPECT(r != var("h", {1, 256}));
    EXPECT(r != var("h"));
}

TEST_CASE(serialize_bounded_var_in_expr)
{
    auto h = var("h", {1, 128});
    auto w = var("w", {1, 256});
    auto e = 2 * h + w - 3;
    auto r = round_trip(e);
    EXPECT(r == e);
    EXPECT(r.eval_uint({{h, 64}, {w, 32}}) == 157);
}

TEST_CASE(serialize_conv_output_with_bounds)
{
    auto h   = var("h", {3, 256});
    auto out = (h - 3) / 2 + 1;
    auto r   = round_trip(out);
    EXPECT(r == out);
    EXPECT(r.eval_uint({{h, 255}}) == 127);
}

TEST_CASE(serialize_comparison_survives_round_trip)
{
    auto h    = var("h", {3, 256});
    auto out  = (h - 3) / 2 + 1;
    auto h2   = round_trip(h);
    auto out2 = round_trip(out);
    EXPECT(out2 < h2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
