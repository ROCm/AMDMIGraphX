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

using migraphx::parser::simple_string_view_skip_parser;

// ---- Base parser tests ----

TEST_CASE(parser_peek_and_advance)
{
    std::string_view sv("ab cd");
    simple_string_view_skip_parser p{sv};
    EXPECT(p.peek_char() == 'a');
    p.advance(1);
    EXPECT(p.peek_char() == 'b');
    p.advance(1);
    EXPECT(p.peek_char() == 'c');
}

TEST_CASE(parser_done)
{
    std::string_view sv("x");
    simple_string_view_skip_parser p{sv};
    EXPECT(not p.done());
    p.advance(1);
    EXPECT(p.done());
}

TEST_CASE(parser_match)
{
    std::string_view sv("hello world");
    simple_string_view_skip_parser p{sv};
    EXPECT(p.match(std::string_view("hello")));
    EXPECT(p.peek_char() == 'w');
    EXPECT(not p.match(std::string_view("xyz")));
    EXPECT(p.peek_char() == 'w');
}

TEST_CASE(parser_parse_while)
{
    std::string_view sv("abc 123");
    simple_string_view_skip_parser p{sv};
    auto letters = p.parse_while([](char c) { return std::isalpha(c); });
    EXPECT(letters == "abc");
    auto digits = p.parse_while([](char c) { return std::isdigit(c); });
    EXPECT(digits == "123");
}

// ---- Combinator tests ----

TEST_CASE(comb_lit_match)
{
    auto comma = migraphx::parser::lit(",");
    std::string_view sv(", rest");
    simple_string_view_skip_parser p{sv};
    auto r = comma(p);
    EXPECT(r.has_value());
    EXPECT(p.peek_char() == 'r');
}

TEST_CASE(comb_lit_no_match)
{
    auto comma = migraphx::parser::lit(",");
    std::string_view sv("xyz");
    simple_string_view_skip_parser p{sv};
    auto r = comma(p);
    EXPECT(not r.has_value());
    EXPECT(p.peek_char() == 'x');
}

TEST_CASE(comb_token)
{
    auto plus = migraphx::parser::token("+");
    std::string_view sv("+ 1");
    simple_string_view_skip_parser p{sv};
    auto r = plus(p);
    EXPECT(r.has_value());
    EXPECT(*r == "+");
}

TEST_CASE(comb_parse_while)
{
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    std::string_view sv("123 abc");
    simple_string_view_skip_parser p{sv};
    auto r = digits(p);
    EXPECT(r.has_value());
    EXPECT(*r == "123");
}

TEST_CASE(comb_parse_while_no_match)
{
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    std::string_view sv("abc");
    simple_string_view_skip_parser p{sv};
    auto r = digits(p);
    EXPECT(not r.has_value());
    EXPECT(p.peek_char() == 'a');
}

TEST_CASE(comb_guard)
{
    auto alpha_guard = migraphx::parser::guard([](char c) { return std::isalpha(c); });
    std::string_view sv("abc");
    simple_string_view_skip_parser p{sv};
    auto r = alpha_guard(p);
    EXPECT(r.has_value());
    EXPECT(p.peek_char() == 'a');

    std::string_view sv2("123");
    simple_string_view_skip_parser p2{sv2};
    auto r2 = alpha_guard(p2);
    EXPECT(not r2.has_value());
}

TEST_CASE(comb_seq_skip_elision)
{
    using migraphx::parser::lit;
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });

    auto paren_num = lit("(") >> digits >> lit(")");
    std::string_view sv("(42)");
    simple_string_view_skip_parser p{sv};
    auto r = paren_num(p);
    EXPECT(r.has_value());
    EXPECT(*r == "42");
}

TEST_CASE(comb_seq_tuple)
{
    auto alpha  = migraphx::parser::parse_while([](char c) { return std::isalpha(c); });
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });

    auto pair = alpha >> digits;
    std::string_view sv("abc 123");
    simple_string_view_skip_parser p{sv};
    auto r = pair(p);
    EXPECT(r.has_value());
    EXPECT(std::get<0>(*r) == "abc");
    EXPECT(std::get<1>(*r) == "123");
}

TEST_CASE(comb_seq_triple_flattened)
{
    auto a = migraphx::parser::parse_while([](char c) { return c == 'a'; });
    auto b = migraphx::parser::parse_while([](char c) { return c == 'b'; });
    auto c = migraphx::parser::parse_while([](char c) { return c == 'c'; });

    auto triple = a >> b >> c;
    std::string_view sv("aa bb cc");
    simple_string_view_skip_parser p{sv};
    auto r = triple(p);
    EXPECT(r.has_value());
    EXPECT(std::get<0>(*r) == "aa");
    EXPECT(std::get<1>(*r) == "bb");
    EXPECT(std::get<2>(*r) == "cc");
}

TEST_CASE(comb_seq_backtrack)
{
    using migraphx::parser::lit;
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });

    auto seq = lit("(") >> digits >> lit(")");
    std::string_view sv("(abc)");
    simple_string_view_skip_parser p{sv};
    auto r = seq(p);
    EXPECT(not r.has_value());
    EXPECT(p.peek_char() == '(');
}

TEST_CASE(comb_alt_same_type)
{
    auto plus  = migraphx::parser::token("+");
    auto minus = migraphx::parser::token("-");
    auto op    = plus | minus;

    std::string_view sv("-");
    simple_string_view_skip_parser p{sv};
    auto r = op(p);
    EXPECT(r.has_value());
    EXPECT(*r == "-");
}

TEST_CASE(comb_alt_no_match)
{
    auto plus  = migraphx::parser::token("+");
    auto minus = migraphx::parser::token("-");
    auto op    = plus | minus;

    std::string_view sv("*");
    simple_string_view_skip_parser p{sv};
    auto r = op(p);
    EXPECT(not r.has_value());
}

TEST_CASE(comb_repeat)
{
    auto word  = migraphx::parser::parse_while([](char c) { return std::isalpha(c); });
    auto words = *word;

    std::string_view sv("foo bar baz 123");
    simple_string_view_skip_parser p{sv};
    auto r = words(p);
    EXPECT(r.has_value());
    EXPECT(r->size() == 3);
    EXPECT(r->at(0) == "foo");
    EXPECT(r->at(1) == "bar");
    EXPECT(r->at(2) == "baz");
}

TEST_CASE(comb_repeat_zero)
{
    auto word  = migraphx::parser::parse_while([](char c) { return std::isalpha(c); });
    auto words = *word;

    std::string_view sv("123");
    simple_string_view_skip_parser p{sv};
    auto r = words(p);
    EXPECT(r.has_value());
    EXPECT(r->empty());
}

TEST_CASE(comb_optional_present)
{
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    auto opt    = -digits;

    std::string_view sv("42");
    simple_string_view_skip_parser p{sv};
    auto r = opt(p);
    EXPECT(r.has_value());
    EXPECT(r->has_value());
    EXPECT(**r == "42");
}

TEST_CASE(comb_optional_absent)
{
    auto digits = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    auto opt    = -digits;

    std::string_view sv("abc");
    simple_string_view_skip_parser p{sv};
    auto r = opt(p);
    EXPECT(r.has_value());
    EXPECT(not r->has_value());
    EXPECT(p.peek_char() == 'a');
}

TEST_CASE(comb_semantic_action)
{
    auto digits    = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    auto to_double = [](std::string_view s) { return std::stod(std::string(s)); };
    auto number    = digits[to_double];

    std::string_view sv("42");
    simple_string_view_skip_parser p{sv};
    auto r = number(p);
    EXPECT(r.has_value());
    EXPECT(*r == 42.0);
}

TEST_CASE(comb_number_list)
{
    using migraphx::parser::lit;
    auto number_s =
        migraphx::parser::parse_while([](char c) { return std::isdigit(c) or c == '.'; });
    auto to_dbl  = [](std::string_view s) { return std::stod(std::string(s)); };
    auto number  = number_s[to_dbl];
    auto numbers = migraphx::parser::separated_by(number, lit(","));

    std::string_view sv("1, 2.5, 3");
    simple_string_view_skip_parser p{sv};
    auto r = numbers(p);
    EXPECT(r.has_value());
    EXPECT(r->size() == 3);
    EXPECT(r->at(0) == 1.0);
    EXPECT(r->at(1) == 2.5);
    EXPECT(r->at(2) == 3.0);
}

TEST_CASE(comb_star_pipe)
{
    auto ident  = migraphx::parser::parse_while([](char c) { return std::isalpha(c); });
    auto number = migraphx::parser::parse_while([](char c) { return std::isdigit(c); });
    auto tokens = *(ident | number);

    std::string_view sv("abc 123 def 456");
    simple_string_view_skip_parser p{sv};
    auto r = tokens(p);
    EXPECT(r.has_value());
    EXPECT(r->size() == 4);
    EXPECT(r->at(0) == "abc");
    EXPECT(r->at(1) == "123");
    EXPECT(r->at(2) == "def");
    EXPECT(r->at(3) == "456");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
