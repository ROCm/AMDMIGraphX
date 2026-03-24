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
#include "test.hpp"

using se = migraphx::symbolic_expr;

// ===================================================================
// Tier 1: Expression construction and canonicalization
// ===================================================================

TEST_CASE(construct_integer)
{
    EXPECT(se(0).to_string() == "0");
    EXPECT(se(1).to_string() == "1");
    EXPECT(se(42).to_string() == "42");
}

TEST_CASE(construct_symbol)
{
    EXPECT(se("H").to_string() == "H");
    EXPECT(se("batch_size").to_string() == "batch_size");
}

TEST_CASE(construct_empty)
{
    se e;
    EXPECT(e.empty());
    EXPECT(e.to_string().empty());
}

TEST_CASE(add_identity)
{
    EXPECT(se("H") + 0 == se("H"));
    EXPECT(0 + se("H") == se("H"));
}

TEST_CASE(add_commutativity)
{
    EXPECT(se("H") + se("W") == se("W") + se("H"));
}

TEST_CASE(add_like_term_folding)
{
    auto r = se("H") + se("H");
    EXPECT(r.to_string() == "2*H");
}

TEST_CASE(add_constant_folding)
{
    EXPECT(se(3) + se(5) == se(8));
    EXPECT(se(0) + se(0) == se(0));
}

TEST_CASE(add_flattening)
{
    auto a = (se("H") + se("W")) + se("C");
    auto b = se("H") + (se("W") + se("C"));
    auto c = (se("C") + se("H")) + se("W");
    EXPECT(a == b);
    EXPECT(b == c);
}

TEST_CASE(add_mixed)
{
    auto r = se("H") + 3 + se("H") + 2;
    EXPECT(r == 2 * se("H") + 5);
    auto r2 = se("H") + se("H");
    EXPECT(r2 + 5 == 2 * se("H") + 5);
}

TEST_CASE(add_cancellation)
{
    auto neg_h = se("-H");
    EXPECT(se("H") + neg_h == se(0));
}

TEST_CASE(sub_identity)
{
    EXPECT(se("H") - 0 == se("H"));
}

TEST_CASE(sub_self)
{
    EXPECT(se("H") - se("H") == se(0));
}

TEST_CASE(sub_constant_folding)
{
    EXPECT(se(10) - se(3) == se(7));
}

TEST_CASE(sub_produces_negation)
{
    EXPECT(se("-H").to_string() == "-H");
    EXPECT((3 - se("H")).to_string() == "-H + 3");
}

TEST_CASE(neg_integer)
{
    auto r = se(0) - 5;
    EXPECT(r == se(-5));
}

TEST_CASE(neg_double_negation)
{
    auto r = 0 - se("-H");
    EXPECT(r == se("H"));
}

TEST_CASE(mul_identity)
{
    EXPECT(se("H") * 1 == se("H"));
    EXPECT(1 * se("H") == se("H"));
}

TEST_CASE(mul_zero)
{
    EXPECT(se("H") * 0 == se(0));
    EXPECT(0 * se("H") == se(0));
}

TEST_CASE(mul_constant_folding)
{
    EXPECT(se(3) * se(7) == se(21));
}

TEST_CASE(mul_commutativity)
{
    EXPECT(se("B") * se("A") == se("A") * se("B"));
}

TEST_CASE(mul_coefficient_accumulation)
{
    auto r = 2 * se("H") * 3;
    EXPECT(r.to_string() == "6*H");
}

TEST_CASE(mul_flattening)
{
    auto a = (se("H") * se("W")) * se("C");
    auto b = se("H") * (se("W") * se("C"));
    auto c = (se("C") * se("H")) * se("W");
    EXPECT(a == b);
    EXPECT(b == c);
}

TEST_CASE(mul_distributive)
{
    auto r = 2 * (se("H") + 1);
    EXPECT(r == 2 * se("H") + 2);
}

TEST_CASE(mul_symbolic_times_add_no_distribution)
{
    auto r = se("N") * (se("H") + 1);
    EXPECT(r != se("N") * se("H") + se("N"));
}

TEST_CASE(fdiv_identity)
{
    EXPECT(se("H") / 1 == se("H"));
}

TEST_CASE(fdiv_constant_folding)
{
    EXPECT(se(7) / se(2) == se(3));
    EXPECT(se(6) / se(3) == se(2));
    EXPECT(se(0) / se(5) == se(0));
}

TEST_CASE(fdiv_exact_coefficient_cancel)
{
    auto r = (6 * se("N")) / 3;
    EXPECT(r.to_string() == "2*N");
}

TEST_CASE(fdiv_non_simplifiable)
{
    auto r = (se("H") - 1) / 2;
    EXPECT(r.to_string() == "(H - 1)/2");
}

TEST_CASE(fdiv_division_by_zero)
{
    EXPECT(test::throws([&] { se("H") / 0; }));
}

TEST_CASE(add_scaled_subtraction)
{
    EXPECT(2 * se("H") - se("H") == se("H"));
    EXPECT(3 * se("H") - 2 * se("H") == se("H"));
    EXPECT(se("H") + se("H") + se("H") == 3 * se("H"));
}

TEST_CASE(add_of_two_adds)
{
    auto r = (se("H") + 1) + (se("H") + 2);
    EXPECT(r == 2 * se("H") + 3);
}

TEST_CASE(sub_strip_constant)
{
    EXPECT((se("H") + 1) - se("H") == se(1));
}

TEST_CASE(sub_of_two_adds)
{
    auto r = (se("H") + 1) - (se("H") + 2);
    EXPECT(r == se(-1));
}

TEST_CASE(mul_zero_propagation)
{
    auto z = se("H") - se("H");
    EXPECT(50 * z == se(0));
}

TEST_CASE(add_chain_constant_cancel)
{
    auto r = se(2) - se("H") - se(2);
    EXPECT(r == se("-H"));
}

TEST_CASE(neg_of_sum_distributes)
{
    auto r = se(-1) * (se("H") + 1);
    EXPECT(r == se("-H") - 1);
}

TEST_CASE(neg_of_product_double)
{
    auto hw  = se("H") * se("W");
    auto neg = 0 - hw;
    EXPECT(neg.to_string() == "-H*W");
    auto dbl = 0 - neg;
    EXPECT(dbl == hw);
}

TEST_CASE(add_compound_product_like_terms)
{
    auto hw = se("H") * se("W");
    auto wh = se("W") * se("H");
    EXPECT(hw + hw == 2 * hw);
    EXPECT(hw + wh == 2 * hw);
    EXPECT(hw + 2 * hw == 3 * hw);
}

TEST_CASE(add_compound_product_cancellation)
{
    auto hw = se("H") * se("W");
    EXPECT(hw - hw == se(0));
}

// X*Y and X cancel pairwise: (X*Y - X) - (X*Y - X) == 0
TEST_CASE(sub_compound_product_mixed)
{
    auto xy = se("X") * se("Y");
    auto r  = xy - se("X") - xy + se("X");
    EXPECT(r == se(0));
}

// Duplicate A*B terms fold even when separated by another term
TEST_CASE(add_multi_term_accumulation)
{
    auto r = se("A") * se("B") + se("C") + se("A") * se("B");
    auto expected = 2 * (se("A") * se("B")) + se("C");
    EXPECT(r == expected);
}

TEST_CASE(fdiv_negative_constant_folding)
{
    EXPECT(se(-7) / se(2) == se(-7 / 2));
    EXPECT(se(-6) / se(3) == se(-2));
    EXPECT(se(7) / se(-2) == se(7 / -2));
}

TEST_CASE(fdiv_large_constants)
{
    EXPECT(se(1000000) / se(1000) == se(1000));
    EXPECT(se(999999) / se(1000) == se(999));
}

// ===================================================================
// Tier 2: Equality and hashing
// ===================================================================

TEST_CASE(eq_different_values)
{
    EXPECT(se("H") + 1 != se("H") + 2);
    EXPECT(se("H") != se("W"));
    EXPECT(se(3) != se(4));
}

TEST_CASE(eq_empty)
{
    EXPECT(se{} == se{});
    EXPECT(se{} != se(0));
    EXPECT(se(0) != se{});
}


// ===================================================================
// Tier 3: Evaluation and substitution
// ===================================================================

TEST_CASE(eval_simple)
{
    EXPECT(se("H").eval({{"H", 26}}) == 26);
    EXPECT(se(42).eval({}) == 42);
}

TEST_CASE(eval_arithmetic)
{
    EXPECT((se("H") - 3).eval({{"H", 26}}) == 23);
    EXPECT((se("H") + 5).eval({{"H", 10}}) == 15);
    EXPECT((2 * se("H")).eval({{"H", 13}}) == 26);
}

TEST_CASE(eval_compound)
{
    auto e = (se("H") - 3) / 2 + 1;
    EXPECT(e.eval({{"H", 26}}) == 12);
    EXPECT(e.eval({{"H", 27}}) == 13);
}

TEST_CASE(eval_multiple_symbols)
{
    auto e = se("N") * se("H");
    EXPECT(e.eval({{"N", 4}, {"H", 26}}) == 104);
}

TEST_CASE(eval_floor_division)
{
    auto e = (se("H") - 1) / 2;
    EXPECT(e.eval({{"H", 7}}) == 3);
    EXPECT(e.eval({{"H", 8}}) == 3);
    EXPECT(e.eval({{"H", 9}}) == 4);
}

TEST_CASE(eval_unbound_throws)
{
    EXPECT(test::throws([&] { se("H").eval({}); }));
    EXPECT(test::throws([&] { (se("H") + se("W")).eval({{"H", 1}}); }));
}

TEST_CASE(eval_integer_expr)
{
    EXPECT(se(0).eval({}) == 0);
    EXPECT(se(100).eval({}) == 100);
}

TEST_CASE(subs_partial)
{
    auto e = se("N") * se("H") + 1;
    auto r = e.subs({{"N", 4}});
    EXPECT(r == 4 * se("H") + 1);
    EXPECT(r.eval({{"H", 10}}) == 41);
}

TEST_CASE(subs_full)
{
    auto e = se("H") + 1;
    auto r = e.subs({{"H", 5}});
    EXPECT(r == se(6));
}

TEST_CASE(subs_none)
{
    auto e = se("H");
    EXPECT(e.subs({}) == se("H"));
}

TEST_CASE(subs_floor_div)
{
    auto e = (se("H") - 1) / 2;
    auto r = e.subs({{"H", 7}});
    EXPECT(r == se(3));
}

// eval() and subs()+eval() must agree on a compound expression
TEST_CASE(subs_eval_cross_validation)
{
    auto e = (se("N") * se("H") - 3) / 2 + 1;
    std::map<std::string, std::size_t> m = {{"N", 4}, {"H", 26}};
    auto via_eval = e.eval(m);
    auto via_subs = e.subs(m).eval({});
    EXPECT(via_eval == via_subs);
}

TEST_CASE(subs_empty)
{
    se e;
    auto r = e.subs({{"H", 5}});
    EXPECT(r.empty());
}

TEST_CASE(subs_creates_like_terms)
{
    auto e = se("H") + se("W");
    auto r = e.subs({{"W", 0}});
    EXPECT(r == se("H"));
}

TEST_CASE(eval_compound_product)
{
    auto e = se("H") * se("W") + 1;
    EXPECT(e.eval({{"H", 3}, {"W", 4}}) == 13);
}

TEST_CASE(eval_negative_intermediate)
{
    auto e = (se("H") - 10) * 2 + 20;
    EXPECT(e.eval({{"H", 3}}) == 6);
}

// ===================================================================
// Tier 4: Printing and parsing
// ===================================================================

TEST_CASE(print_atoms)
{
    EXPECT(se(42).to_string() == "42");
    EXPECT(se("H").to_string() == "H");
    EXPECT(se(0).to_string() == "0");
    EXPECT(se(-3).to_string() == "-3");
}

TEST_CASE(print_add)
{
    EXPECT((se("H") + 1).to_string() == "H + 1");
    EXPECT((se("H") - 3).to_string() == "H - 3");
}

TEST_CASE(print_mul)
{
    EXPECT((2 * se("H")).to_string() == "2*H");
    auto r = se("A") * se("B");
    EXPECT(r.to_string() == "A*B");
}

TEST_CASE(print_fdiv_parens)
{
    auto r = (se("H") - 1) / 2;
    EXPECT(r.to_string() == "(H - 1)/2");
}

TEST_CASE(print_compound)
{
    auto r = (se("H") - 3) / 2 + 1;
    EXPECT(r.to_string() == "(H - 3)/2 + 1");
}

TEST_CASE(parse_atoms)
{
    EXPECT(se("42") == se(42));
    EXPECT(se("H") == se("H"));
}

TEST_CASE(parse_arithmetic)
{
    auto r = se("H + 1");
    EXPECT(r == se("H") + 1);

    auto r2 = se("H - 3");
    EXPECT(r2 == se("H") - 3);

    auto r3 = se("2*H");
    EXPECT(r3 == 2 * se("H"));
}

TEST_CASE(parse_precedence)
{
    auto r = se("H + 1 * 2");
    EXPECT(r == se("H") + 2);
}

TEST_CASE(parse_parentheses)
{
    auto r = se("(H + 1) * 2");
    EXPECT(r == 2 * (se("H") + 1));
}

TEST_CASE(parse_division)
{
    auto r = se("(H - 1)/2");
    EXPECT(r == (se("H") - 1) / 2);
}

TEST_CASE(parse_unary_minus)
{
    EXPECT(se("-H") == se("-H"));
    EXPECT(se("-H").to_string() == "-H");
    EXPECT(se("-(H + 1)") == se("-H") - 1);
}

// Legacy floor() wrapper is accepted by parser and treated as no-op
TEST_CASE(parse_floor_backward_compat)
{
    auto a = se("floor((H-1)/2)");
    auto b = se("(H-1)/2");
    EXPECT(a == b);

    auto c = se("floor((H-1)/2) + 1");
    auto d = (se("H") - 1) / 2 + 1;
    EXPECT(c == d);
}

TEST_CASE(parse_whitespace_tolerance)
{
    EXPECT(se("  H  +  1  ") == se("H + 1"));
    EXPECT(se("H+1") == se("H + 1"));
}

TEST_CASE(print_negative_mul_coefficient)
{
    auto r = 0 - 3 * se("H");
    EXPECT(r.to_string() == "-3*H");
}

TEST_CASE(print_multi_symbol_product)
{
    auto r = se("H") * se("W");
    auto s = r.to_string();
    EXPECT(s == "H*W" or s == "W*H");
    EXPECT(se("H*W") == se("W*H"));
}

TEST_CASE(print_compound_expression)
{
    auto r = 2 * (se("H") * se("W")) + se("C") - 1;
    auto s = r.to_string();
    EXPECT(se(s) == r);
}

TEST_CASE(parse_compound_mul)
{
    auto r = se("2*H*W");
    EXPECT(r == 2 * se("H") * se("W"));
}

TEST_CASE(print_parse_round_trip)
{
    std::vector<se> exprs = {
        se("H"),
        se("H") + 1,
        2 * se("H") - 3,
        (se("H") - 3) / 2 + 1,
        se("N") * se("C") * se("H") * se("W"),
        (se("H") - 1) / 2,
    };
    for(const auto& e : exprs)
    {
        auto s        = e.to_string();
        auto reparsed = se(s);
        EXPECT(reparsed == e);
    }
}

// ===================================================================
// Tier 6: Edge cases and robustness
// ===================================================================

// 5 levels of (e-1)/2: simulates repeated pooling/conv stride reduction
TEST_CASE(edge_deeply_nested)
{
    auto e = se("H");
    for(int i = 0; i < 5; ++i)
        e = (e - 1) / 2;
    EXPECT(e.eval({{"H", 255}}) == 7);
}

TEST_CASE(edge_many_symbols)
{
    auto e = se("A") + se("B") + se("C") + se("D") + se("E");
    EXPECT(e.eval({{"A", 1}, {"B", 2}, {"C", 3}, {"D", 4}, {"E", 5}}) == 15);
}

TEST_CASE(edge_neg_one_coefficient)
{
    EXPECT(se("-H").to_string() == "-H");
    EXPECT(se("-H") + se("H") == se(0));
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
    auto r1 = se("H") + empty;
    EXPECT(not r1.empty());

    auto r2 = empty + se("H");
    EXPECT(not r2.empty());
}

TEST_CASE(edge_large_coefficients)
{
    auto r = 1000000 * se("H");
    EXPECT(r.eval({{"H", 1000000}}) == 1000000000000ULL);
}

// Incrementally adding H ten times must fold to 11*H
TEST_CASE(edge_chained_operations)
{
    auto e = se("H");
    for(int i = 0; i < 10; ++i)
        e = e + se("H");
    EXPECT(e == 11 * se("H"));
}


TEST_CASE(edge_repeated_parse)
{
    for(int i = 0; i < 10; ++i)
    {
        auto r = se("(H - 3)/2 + 1");
        EXPECT(r == (se("H") - 3) / 2 + 1);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
