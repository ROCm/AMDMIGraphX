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

using migraphx::parser::action;
using migraphx::parser::simple_string_view_skip_parser;

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

TEST_CASE(parser_try_parse)
{
    std::string_view sv("hello world");
    simple_string_view_skip_parser p{sv};
    bool matched = p.try_parse([](auto& q) { q.match(std::string_view("hello")); });
    EXPECT(matched);
    EXPECT(p.peek_char() == 'w');

    bool missed = p.try_parse([](auto& q) { q.match(std::string_view("xyz")); });
    EXPECT(not missed);
    EXPECT(p.peek_char() == 'w');
}

TEST_CASE(parser_parse_first_of)
{
    std::string_view sv("42");
    simple_string_view_skip_parser p{sv};
    auto result = p.parse_first_of(
        [](auto& q) -> std::string {
            if(not std::isalpha(q.peek_char()))
                return {};
            return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
        },
        [](auto& q) -> std::string {
            if(not std::isdigit(q.peek_char()))
                return {};
            return std::string(q.parse_while([](char c) { return std::isdigit(c); }));
        });
    EXPECT(result == "42");
}

TEST_CASE(parser_parse_first_of_first_match)
{
    std::string_view sv("abc");
    simple_string_view_skip_parser p{sv};
    auto result = p.parse_first_of(
        [](auto& q) -> std::string {
            if(not std::isalpha(q.peek_char()))
                return {};
            return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
        },
        [](auto& q) -> std::string {
            if(not std::isdigit(q.peek_char()))
                return {};
            return std::string(q.parse_while([](char c) { return std::isdigit(c); }));
        });
    EXPECT(result == "abc");
}

TEST_CASE(parser_parse_repeat)
{
    std::string_view sv("a b c .");
    simple_string_view_skip_parser p{sv};
    auto results = p.parse_repeat([](auto& q) -> std::string {
        if(not std::isalpha(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
    });
    EXPECT(results.size() == 3);
    EXPECT(results[0] == "a");
    EXPECT(results[1] == "b");
    EXPECT(results[2] == "c");
    EXPECT(p.peek_char() == '.');
}

TEST_CASE(parser_parse_repeat_empty)
{
    std::string_view sv("123");
    simple_string_view_skip_parser p{sv};
    auto results = p.parse_repeat([](auto& q) -> std::string {
        if(not std::isalpha(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
    });
    EXPECT(results.empty());
    EXPECT(p.peek_char() == '1');
}

TEST_CASE(parser_action_pipe)
{
    auto parse_alpha  = action([](auto& q) -> std::string {
        if(not std::isalpha(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
    });
    auto parse_digits = action([](auto& q) -> std::string {
        if(not std::isdigit(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isdigit(c); }));
    });
    auto parse_token  = parse_alpha | parse_digits;

    std::string_view sv("123");
    simple_string_view_skip_parser p{sv};
    auto result = parse_token(p);
    EXPECT(result == "123");

    std::string_view sv2("abc");
    simple_string_view_skip_parser p2{sv2};
    auto result2 = parse_token(p2);
    EXPECT(result2 == "abc");
}

TEST_CASE(parser_action_star)
{
    auto parse_word  = action([](auto& q) -> std::string {
        if(not std::isalpha(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isalpha(c); }));
    });
    auto parse_words = *parse_word;

    std::string_view sv("foo bar baz 123");
    simple_string_view_skip_parser p{sv};
    auto results = parse_words(p);
    EXPECT(results.size() == 3);
    EXPECT(results[0] == "foo");
    EXPECT(results[1] == "bar");
    EXPECT(results[2] == "baz");
}

TEST_CASE(parser_action_star_pipe)
{
    auto parse_ident  = action([](auto& q) -> std::string {
        if(not std::isalpha(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isalnum(c); }));
    });
    auto parse_number = action([](auto& q) -> std::string {
        if(not std::isdigit(q.peek_char()))
            return {};
        return std::string(q.parse_while([](char c) { return std::isdigit(c); }));
    });
    auto parse_token  = parse_number | parse_ident;
    auto parse_tokens = *parse_token;

    std::string_view sv("123 abc 456 def");
    simple_string_view_skip_parser p{sv};
    auto results = parse_tokens(p);
    EXPECT(results.size() == 4);
    EXPECT(results[0] == "123");
    EXPECT(results[1] == "abc");
    EXPECT(results[2] == "456");
    EXPECT(results[3] == "def");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
