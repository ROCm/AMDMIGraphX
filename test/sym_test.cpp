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
    EXPECT(var("H").to_string() == "H");
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
    auto H = var("H");
    EXPECT(H + 0 == H);
    EXPECT(0 + H == H);
}

TEST_CASE(add_commutativity)
{
    auto H = var("H"), W = var("W");
    EXPECT(H + W == W + H);
}

TEST_CASE(add_like_term_folding)
{
    auto H = var("H");
    auto r = H + H;
    EXPECT(r == 2 * H);
}

TEST_CASE(add_constant_folding)
{
    EXPECT(lit(3) + lit(5) == lit(8));
    EXPECT(lit(0) + lit(0) == lit(0));
}

TEST_CASE(add_flattening)
{
    auto H = var("H"), W = var("W"), C = var("C");
    auto a = (H + W) + C;
    auto b = H + (W + C);
    auto c = (C + H) + W;
    EXPECT(a == b);
    EXPECT(b == c);
}

TEST_CASE(add_mixed)
{
    auto H = var("H");
    auto r = H + 3 + H + 2;
    EXPECT(r == 2 * H + 5);
    auto r2 = H + H;
    EXPECT(r2 + 5 == 2 * H + 5);
}

TEST_CASE(add_cancellation)
{
    auto H = var("H");
    EXPECT(H + (-1 * H) == lit(0));
}

TEST_CASE(sub_identity)
{
    auto H = var("H");
    EXPECT(H - 0 == H);
}

TEST_CASE(sub_self)
{
    auto H = var("H");
    EXPECT(H - H == lit(0));
}

TEST_CASE(sub_constant_folding) { EXPECT(lit(10) - lit(3) == lit(7)); }

TEST_CASE(sub_produces_negation)
{
    auto H = var("H");
    EXPECT(-1 * H == lit(0) - H);
    EXPECT(3 - H == lit(0) - H + 3);
}

TEST_CASE(neg_integer)
{
    auto r = lit(0) - 5;
    EXPECT(r == lit(-5));
}

TEST_CASE(neg_double_negation)
{
    auto H = var("H");
    auto r = 0 - (-1 * H);
    EXPECT(r == H);
}

TEST_CASE(mul_identity)
{
    auto H = var("H");
    EXPECT(H * 1 == H);
    EXPECT(1 * H == H);
}

TEST_CASE(mul_zero)
{
    auto H = var("H");
    EXPECT(H * 0 == lit(0));
    EXPECT(0 * H == lit(0));
}

TEST_CASE(mul_constant_folding) { EXPECT(lit(3) * lit(7) == lit(21)); }

TEST_CASE(mul_commutativity)
{
    auto A = var("A"), B = var("B");
    EXPECT(B * A == A * B);
}

TEST_CASE(mul_coefficient_accumulation)
{
    auto H = var("H");
    auto r = 2 * H * 3;
    EXPECT(r == 6 * H);
}

TEST_CASE(mul_flattening)
{
    auto H = var("H"), W = var("W"), C = var("C");
    auto a = (H * W) * C;
    auto b = H * (W * C);
    auto c = (C * H) * W;
    EXPECT(a == b);
    EXPECT(b == c);
}

TEST_CASE(mul_distributive)
{
    auto H = var("H");
    auto r = 2 * (H + 1);
    EXPECT(r == 2 * H + 2);
}

TEST_CASE(fdiv_identity)
{
    auto H = var("H");
    EXPECT(H / 1 == H);
}

TEST_CASE(fdiv_constant_folding)
{
    EXPECT(lit(7) / lit(2) == lit(3));
    EXPECT(lit(6) / lit(3) == lit(2));
    EXPECT(lit(0) / lit(5) == lit(0));
}

TEST_CASE(fdiv_exact_coefficient_cancel)
{
    auto N = var("N");
    auto r = (6 * N) / 3;
    EXPECT(r == 2 * N);
}

TEST_CASE(fdiv_non_simplifiable)
{
    auto H = var("H");
    auto r = (H - 1) / 2;
    EXPECT(r == (H - 1) / 2);
}

TEST_CASE(fdiv_division_by_zero)
{
    EXPECT(test::throws([&] { var("H") / 0; }));
}

TEST_CASE(add_scaled_subtraction)
{
    auto H = var("H");
    EXPECT(2 * H - H == H);
    EXPECT(3 * H - 2 * H == H);
    EXPECT(H + H + H == 3 * H);
}

TEST_CASE(add_of_two_adds)
{
    auto H = var("H");
    auto r = (H + 1) + (H + 2);
    EXPECT(r == 2 * H + 3);
}

TEST_CASE(sub_strip_constant)
{
    auto H = var("H");
    EXPECT((H + 1) - H == lit(1));
}

TEST_CASE(sub_of_two_adds)
{
    auto H = var("H");
    auto r = (H + 1) - (H + 2);
    EXPECT(r == lit(-1));
}

TEST_CASE(mul_zero_propagation)
{
    auto H = var("H");
    auto z = H - H;
    EXPECT(50 * z == lit(0));
}

TEST_CASE(add_chain_constant_cancel)
{
    auto H = var("H");
    auto r = lit(2) - H - lit(2);
    EXPECT(r == -1 * H);
}

TEST_CASE(neg_of_sum_distributes)
{
    auto H = var("H");
    auto r = lit(-1) * (H + 1);
    EXPECT(r == -1 * H - 1);
}

TEST_CASE(neg_of_product_double)
{
    auto hw  = var("H") * var("W");
    auto neg = 0 - hw;
    EXPECT(neg == lit(0) - hw);
    auto dbl = 0 - neg;
    EXPECT(dbl == hw);
}

TEST_CASE(add_compound_product_like_terms)
{
    auto H = var("H"), W = var("W");
    auto hw = H * W;
    auto wh = W * H;
    EXPECT(hw + hw == 2 * hw);
    EXPECT(hw + wh == 2 * hw);
    EXPECT(hw + 2 * hw == 3 * hw);
}

TEST_CASE(add_compound_product_cancellation)
{
    auto hw = var("H") * var("W");
    EXPECT(hw - hw == lit(0));
}

// X*Y and X cancel pairwise: (X*Y - X) - (X*Y - X) == 0
TEST_CASE(sub_compound_product_mixed)
{
    auto X = var("X"), Y = var("Y");
    auto xy = X * Y;
    auto r  = xy - X - xy + X;
    EXPECT(r == lit(0));
}

// Duplicate A*B terms fold even when separated by another term
TEST_CASE(add_multi_term_accumulation)
{
    auto A = var("A"), B = var("B"), C = var("C");
    auto r        = A * B + C + A * B;
    auto expected = 2 * (A * B) + C;
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
    auto H = var("H"), W = var("W");
    EXPECT(H + 1 != H + 2);
    EXPECT(H != W);
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
    auto H = var("H");
    EXPECT(H.eval({{H, 26}}) == 26);
    EXPECT(lit(42).eval({}) == 42);
}

TEST_CASE(eval_arithmetic)
{
    auto H = var("H");
    EXPECT((H - 3).eval({{H, 26}}) == 23);
    EXPECT((H + 5).eval({{H, 10}}) == 15);
    EXPECT((2 * H).eval({{H, 13}}) == 26);
}

TEST_CASE(eval_compound)
{
    auto H = var("H");
    auto e = (H - 3) / 2 + 1;
    EXPECT(e.eval({{H, 26}}) == 12);
    EXPECT(e.eval({{H, 27}}) == 13);
}

TEST_CASE(eval_multiple_symbols)
{
    auto N = var("N"), H = var("H");
    auto e = N * H;
    EXPECT(e.eval({{N, 4}, {H, 26}}) == 104);
}

TEST_CASE(eval_floor_division)
{
    auto H = var("H");
    auto e = (H - 1) / 2;
    EXPECT(e.eval({{H, 7}}) == 3);
    EXPECT(e.eval({{H, 8}}) == 3);
    EXPECT(e.eval({{H, 9}}) == 4);
}

TEST_CASE(eval_unbound_throws)
{
    auto H = var("H"), W = var("W");
    EXPECT(test::throws([&] { H.eval({}); }));
    EXPECT(test::throws([&] { (H + W).eval({{H, 1}}); }));
}

TEST_CASE(eval_integer_expr)
{
    EXPECT(lit(0).eval({}) == 0);
    EXPECT(lit(100).eval({}) == 100);
}

TEST_CASE(subs_partial)
{
    auto N = var("N"), H = var("H");
    auto e = N * H + 1;
    auto r = e.subs({{N, lit(4)}});
    EXPECT(r == 4 * H + 1);
    EXPECT(r.eval({{H, 10}}) == 41);
}

TEST_CASE(subs_full)
{
    auto H = var("H");
    auto e = H + 1;
    auto r = e.subs({{H, lit(5)}});
    EXPECT(r == lit(6));
}

TEST_CASE(subs_none)
{
    auto H = var("H");
    EXPECT(H.subs({}) == H);
}

TEST_CASE(subs_floor_div)
{
    auto H = var("H");
    auto e = (H - 1) / 2;
    auto r = e.subs({{H, lit(7)}});
    EXPECT(r == lit(3));
}

// eval() and subs()+eval() must agree on a compound expression
TEST_CASE(subs_eval_cross_validation)
{
    auto N = var("N"), H = var("H");
    auto e                                       = (N * H - 3) / 2 + 1;
    std::unordered_map<se, std::size_t> eval_map = {{N, 4}, {H, 26}};
    std::unordered_map<se, se> subs_map          = {{N, lit(4)}, {H, lit(26)}};
    auto via_eval                                = e.eval(eval_map);
    auto via_subs                                = e.subs(subs_map).eval({});
    EXPECT(via_eval == via_subs);
}

TEST_CASE(subs_empty)
{
    se e;
    auto r = e.subs({{var("H"), lit(5)}});
    EXPECT(r.empty());
}

TEST_CASE(subs_creates_like_terms)
{
    auto H = var("H"), W = var("W");
    auto e = H + W;
    auto r = e.subs({{W, lit(0)}});
    EXPECT(r == H);
}

TEST_CASE(subs_with_expression)
{
    auto H = var("H"), W = var("W");
    auto e = 2 * H + 1;
    auto r = e.subs({{H, W + 3}});
    EXPECT(r == 2 * W + 7);
}

TEST_CASE(subs_symbol_for_symbol)
{
    auto H = var("H"), W = var("W");
    auto e = H * H + 1;
    auto r = e.subs({{H, W}});
    EXPECT(r == W * W + 1);
}

TEST_CASE(subs_compound_expression)
{
    auto N = var("N"), H = var("H"), W = var("W");
    auto e = (N * H + W - 3) / 2;
    auto r = e.subs({{H, 2 * W + 1}, {N, W - 1}});
    // N*H => (W-1)*(2*W+1) = 2*W^2 - W - 1
    // N*H + W - 3 => 2*W^2 - 2*W - 4 + W = 2*W^2 - 2
    // Verify by evaluating with W=5: (W-1)*(2*W+1) + W - 3 = 4*11 + 5 - 3 = 46, 46/2 = 23
    EXPECT(r.eval({{W, 5}}) == 23);
    // Also verify the original expression with direct values agrees
    EXPECT(e.eval({{N, 4}, {H, 11}, {W, 5}}) == 23);
}

TEST_CASE(eval_compound_product)
{
    auto H = var("H"), W = var("W");
    auto e = H * W + 1;
    EXPECT(e.eval({{H, 3}, {W, 4}}) == 13);
}

TEST_CASE(eval_negative_intermediate)
{
    auto H = var("H");
    auto e = (H - 10) * 2 + 20;
    EXPECT(e.eval({{H, 3}}) == 6);
}

// ===================================================================
// Tier 4: Printing and parsing
// ===================================================================

TEST_CASE(print_atoms)
{
    EXPECT(lit(42).to_string() == "42");
    EXPECT(var("H").to_string() == "H");
    EXPECT(lit(0).to_string() == "0");
    EXPECT(lit(-3).to_string() == "-3");
}

TEST_CASE(print_add)
{
    auto H = var("H");
    EXPECT((H + 1).to_string() == "H + 1");
    EXPECT((H - 3).to_string() == "H - 3");
}

TEST_CASE(print_mul)
{
    EXPECT((2 * var("H")).to_string() == "2*H");
    auto r = var("A") * var("B");
    EXPECT(r.to_string() == "A*B");
}

TEST_CASE(print_fdiv_parens)
{
    auto r = (var("H") - 1) / 2;
    EXPECT(r.to_string() == "(H - 1)/2");
}

TEST_CASE(print_compound)
{
    auto r = (var("H") - 3) / 2 + 1;
    EXPECT(r.to_string() == "(H - 3)/2 + 1");
}

TEST_CASE(parse_atoms)
{
    EXPECT(parse("42") == lit(42));
    EXPECT(parse("H") == var("H"));
}

TEST_CASE(parse_arithmetic)
{
    auto H = var("H");
    auto r = parse("H + 1");
    EXPECT(r == H + 1);

    auto r2 = parse("H - 3");
    EXPECT(r2 == H - 3);

    auto r3 = parse("2*H");
    EXPECT(r3 == 2 * H);
}

TEST_CASE(parse_precedence)
{
    auto r = parse("H + 1 * 2");
    EXPECT(r == var("H") + 2);
}

TEST_CASE(parse_parentheses)
{
    auto r = parse("(H + 1) * 2");
    EXPECT(r == 2 * (var("H") + 1));
}

TEST_CASE(parse_division)
{
    auto r = parse("(H - 1)/2");
    EXPECT(r == (var("H") - 1) / 2);
}

TEST_CASE(parse_unary_minus)
{
    auto H = var("H");
    EXPECT(parse("-H") == -1 * H);
    EXPECT(parse("-H").to_string() == "-H");
    EXPECT(parse("-(H + 1)") == -1 * H - 1);
}

// Legacy floor() wrapper is accepted by parser and treated as no-op
TEST_CASE(parse_floor_backward_compat)
{
    auto a = parse("floor((H-1)/2)");
    auto b = parse("(H-1)/2");
    EXPECT(a == b);

    auto c = parse("floor((H-1)/2) + 1");
    auto d = (var("H") - 1) / 2 + 1;
    EXPECT(c == d);
}

TEST_CASE(parse_whitespace_tolerance)
{
    EXPECT(parse("  H  +  1  ") == parse("H + 1"));
    EXPECT(parse("H+1") == parse("H + 1"));
}

TEST_CASE(print_negative_mul_coefficient)
{
    auto r = 0 - 3 * var("H");
    EXPECT(r.to_string() == "-3*H");
}

TEST_CASE(print_multi_symbol_product)
{
    auto r = var("H") * var("W");
    auto s = r.to_string();
    EXPECT(s == "H*W" or s == "W*H");
    EXPECT(parse("H*W") == parse("W*H"));
}

TEST_CASE(print_compound_expression)
{
    auto r = 2 * (var("H") * var("W")) + var("C") - 1;
    auto s = r.to_string();
    EXPECT(parse(s) == r);
}

TEST_CASE(parse_compound_mul)
{
    auto r = parse("2*H*W");
    EXPECT(r == 2 * var("H") * var("W"));
}

TEST_CASE(print_parse_round_trip)
{
    auto H = var("H"), N = var("N"), C = var("C"), W = var("W");
    std::vector<se> exprs = {
        H,
        H + 1,
        2 * H - 3,
        (H - 3) / 2 + 1,
        N * C * H * W,
        (H - 1) / 2,
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
    auto H = var("H");
    se e   = H;
    for(int i = 0; i < 5; ++i)
        e = (e - 1) / 2;
    EXPECT(e.eval({{H, 255}}) == 7);
}

TEST_CASE(edge_many_symbols)
{
    auto A = var("A"), B = var("B"), C = var("C"), D = var("D"), E = var("E");
    auto e = A + B + C + D + E;
    EXPECT(e.eval({{A, 1}, {B, 2}, {C, 3}, {D, 4}, {E, 5}}) == 15);
}

TEST_CASE(edge_neg_one_coefficient)
{
    auto H = var("H");
    EXPECT(-1 * H == lit(0) - H);
    EXPECT(-1 * H + H == lit(0));
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
    auto H  = var("H");
    auto r1 = H + empty;
    EXPECT(not r1.empty());

    auto r2 = empty + H;
    EXPECT(not r2.empty());
}

TEST_CASE(edge_large_coefficients)
{
    auto H = var("H");
    auto r = 1000000 * H;
    EXPECT(r.eval({{H, 1000000}}) == 1000000000000ULL);
}

// Incrementally adding H ten times must fold to 11*H
TEST_CASE(edge_chained_operations)
{
    auto H = var("H");
    auto e = H;
    for(int i = 0; i < 10; ++i)
        e = e + H;
    EXPECT(e == 11 * H);
}

TEST_CASE(edge_repeated_parse)
{
    auto H = var("H");
    for(int i = 0; i < 10; ++i)
    {
        auto r = parse("(H - 3)/2 + 1");
        EXPECT(r == (H - 3) / 2 + 1);
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
    auto H = var("H");
    EXPECT(round_trip(H) == H);
}

TEST_CASE(serialize_add)
{
    auto H = var("H");
    auto e = 2 * H + 3;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_mul)
{
    auto H = var("H"), W = var("W");
    auto e = H * W;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_fdiv)
{
    auto H = var("H");
    auto e = (H - 1) / 2;
    EXPECT(round_trip(e) == e);
}

TEST_CASE(serialize_compound)
{
    auto N = var("N"), H = var("H"), W = var("W");
    auto e = (N * H * W + 3) / 2 - 1;
    EXPECT(round_trip(e) == e);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
