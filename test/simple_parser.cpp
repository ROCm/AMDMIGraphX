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
#include <migraphx/simple_parser.hpp>
#include <test.hpp>

using migraphx::parser::guard;
using migraphx::parser::lit;
using migraphx::parser::parse;
using migraphx::parser::parse_state;
using migraphx::parser::parse_while;
using migraphx::parser::separated_by;
using migraphx::parser::token;

// ---- lit / token ----

TEST_CASE(comb_lit_match)
{
    parse_state ps(", rest");
    auto r = lit(",")(ps);
    EXPECT(r.has_value());
    EXPECT(ps.peek_char() == 'r');
}

TEST_CASE(comb_lit_no_match)
{
    parse_state ps("xyz");
    auto r = lit(",")(ps);
    EXPECT(not r.has_value());
    EXPECT(ps.peek_char() == 'x');
}

TEST_CASE(comb_token)
{
    auto r = parse("++", token("++"));
    EXPECT(r == "++");
}

// ---- parse_while / guard ----

TEST_CASE(comb_parse_while)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto r      = parse("123", digits);
    EXPECT(r == "123");
}

TEST_CASE(comb_parse_while_no_match)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    parse_state ps("abc");
    auto r = digits(ps);
    EXPECT(not r.has_value());
    EXPECT(ps.peek_char() == 'a');
}

TEST_CASE(comb_guard)
{
    auto alpha_guard = guard([](char c) { return std::isalpha(c); });
    parse_state ps("abc");
    auto r = alpha_guard(ps);
    EXPECT(r.has_value());
    EXPECT(ps.peek_char() == 'a');

    parse_state ps2("123");
    auto r2 = alpha_guard(ps2);
    EXPECT(not r2.has_value());
}

// ---- >> : sequence ----

TEST_CASE(comb_seq_skip_elision)
{
    auto digits   = parse_while([](char c) { return std::isdigit(c); });
    auto paren_num = lit("(") >> digits >> lit(")");
    auto r        = parse("(42)", paren_num);
    EXPECT(r == "42");
}

TEST_CASE(comb_seq_tuple)
{
    auto alpha  = parse_while([](char c) { return std::isalpha(c); });
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto pair   = alpha >> digits;
    auto r      = parse("abc 123", pair);
    EXPECT(std::get<0>(r) == "abc");
    EXPECT(std::get<1>(r) == "123");
}

TEST_CASE(comb_seq_triple_flattened)
{
    auto a      = parse_while([](char c) { return c == 'a'; });
    auto b      = parse_while([](char c) { return c == 'b'; });
    auto c      = parse_while([](char c) { return c == 'c'; });
    auto triple = a >> b >> c;
    auto r      = parse("aa bb cc", triple);
    EXPECT(std::get<0>(r) == "aa");
    EXPECT(std::get<1>(r) == "bb");
    EXPECT(std::get<2>(r) == "cc");
}

TEST_CASE(comb_seq_backtrack)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto seq    = lit("(") >> digits >> lit(")");
    parse_state ps("(abc)");
    auto r = seq(ps);
    EXPECT(not r.has_value());
    EXPECT(ps.peek_char() == '(');
}

// ---- | : alternative ----

TEST_CASE(comb_alt_same_type)
{
    auto op = token("+") | token("-");
    auto r  = parse("-", op);
    EXPECT(r == "-");
}

TEST_CASE(comb_alt_no_match)
{
    auto op = token("+") | token("-");
    parse_state ps("*");
    auto r = op(ps);
    EXPECT(not r.has_value());
}

// ---- * : repetition ----

TEST_CASE(comb_repeat)
{
    auto word  = parse_while([](char c) { return std::isalpha(c); });
    auto words = *word;
    auto r     = parse("foo bar baz", words);
    EXPECT(r.size() == 3);
    EXPECT(r[0] == "foo");
    EXPECT(r[1] == "bar");
    EXPECT(r[2] == "baz");
}

TEST_CASE(comb_repeat_zero)
{
    auto word  = parse_while([](char c) { return std::isalpha(c); });
    auto words = *word;
    parse_state ps("123");
    auto r = words(ps);
    EXPECT(r.has_value());
    EXPECT(r->empty());
}

// ---- - : optional ----

TEST_CASE(comb_optional_present)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto opt    = -digits;
    auto r      = parse("42", opt);
    EXPECT(r.has_value());
    EXPECT(*r == "42");
}

TEST_CASE(comb_optional_absent)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto opt    = -digits;
    parse_state ps("abc");
    auto r = opt(ps);
    EXPECT(r.has_value());
    EXPECT(not r->has_value());
    EXPECT(ps.peek_char() == 'a');
}

// ---- [] : semantic action ----

TEST_CASE(comb_semantic_action)
{
    auto digits    = parse_while([](char c) { return std::isdigit(c); });
    auto to_double = [](std::string_view s) { return std::stod(std::string(s)); };
    auto number    = digits[to_double];
    auto r         = parse("42", number);
    EXPECT(r == 42.0);
}

// ---- separated_by ----

TEST_CASE(comb_number_list)
{
    auto number_s = parse_while([](char c) { return std::isdigit(c) or c == '.'; });
    auto to_dbl   = [](std::string_view s) { return std::stod(std::string(s)); };
    auto number   = number_s[to_dbl];
    auto numbers  = separated_by(number, lit(","));
    auto r        = parse("1, 2.5, 3", numbers);
    EXPECT(r.size() == 3);
    EXPECT(r[0] == 1.0);
    EXPECT(r[1] == 2.5);
    EXPECT(r[2] == 3.0);
}

// ---- composition: *(p1 | p2) ----

TEST_CASE(comb_star_pipe)
{
    auto ident  = parse_while([](char c) { return std::isalpha(c); });
    auto number = parse_while([](char c) { return std::isdigit(c); });
    auto tokens = *(ident | number);
    auto r      = parse("abc 123 def 456", tokens);
    EXPECT(r.size() == 4);
    EXPECT(r[0] == "abc");
    EXPECT(r[1] == "123");
    EXPECT(r[2] == "def");
    EXPECT(r[3] == "456");
}

// ---- parse function ----

TEST_CASE(parse_full_match)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    auto r      = parse("42", digits);
    EXPECT(r == "42");
}

TEST_CASE(parse_throws_on_remaining)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    bool threw  = false;
    try
    {
        parse("42 abc", digits);
    }
    catch(const migraphx::exception&)
    {
        threw = true;
    }
    EXPECT(threw);
}

TEST_CASE(parse_throws_on_no_match)
{
    auto digits = parse_while([](char c) { return std::isdigit(c); });
    bool threw  = false;
    try
    {
        parse("abc", digits);
    }
    catch(const migraphx::exception&)
    {
        threw = true;
    }
    EXPECT(threw);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
