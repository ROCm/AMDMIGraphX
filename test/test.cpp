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
#include <test.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

static bool glob_match(const std::string& input, const std::string& pattern)
{
    return test::glob_match(input.begin(), input.end(), pattern.begin(), pattern.end());
}

TEST_CASE(globbing)
{
    EXPECT(not glob_match("ab", "a"));
    EXPECT(not glob_match("ba", "a"));
    EXPECT(not glob_match("bac", "a"));
    EXPECT(glob_match("ab", "ab"));

    // Star loop
    EXPECT(glob_match("/foo/bar/baz/blig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("/foo/bar/baz/xlig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*b"));
    EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
                      "a*a*a*a*a*a*a*a**a*a*a*a*b"));
    EXPECT(glob_match("aabaabaab", "a*"));
    EXPECT(glob_match("aabaabaab", "a*b*ab"));
    EXPECT(glob_match("aabaabaab", "a*baab"));
    EXPECT(glob_match("aabaabaab", "aa*"));
    EXPECT(glob_match("aabaabaab", "aaba*"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("abcdd", "*d"));
    EXPECT(glob_match("abcdd", "*d*"));
    EXPECT(glob_match("daaadabadmanda", "da*da*da*"));
    EXPECT(glob_match("mississippi", "m*issip*"));
    EXPECT(glob_match("abc", "ab*c"));

    // Repeated star
    EXPECT(glob_match("aabaabqqbaab", "a****baab"));
    EXPECT(glob_match("abcdd", "***d"));
    EXPECT(glob_match("abcdd", "***d****"));
    EXPECT(not glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "a**z"));

    // Single wildcard
    EXPECT(glob_match("abc", "a?c"));
    EXPECT(not glob_match("abc", "ab?c"));

    // Special characters
    EXPECT(glob_match("test.foo[gpu]", "test.foo[gpu]"));
    EXPECT(glob_match("test.foo[gpu]", "test.foo[*]"));
    EXPECT(glob_match("test.foo[gpu]", "*[*"));

    EXPECT(glob_match("test.foo(gpu)", "test.foo(gpu)"));
    EXPECT(glob_match("test.foo(gpu)", "test.foo(*)"));
    EXPECT(glob_match("test.foo(gpu)", "*(*"));

    EXPECT(not glob_match("test.foog", "test.foo[gpu]"));
    EXPECT(not glob_match("test.foogpu", "test.foo[gpu]"));
    EXPECT(not glob_match("test_foo", "test.foo"));
}

// Tests for as_string / print_stream
TEST_CASE(as_string_basic_types)
{
    EXPECT(test::as_string(42) == "42");
    EXPECT(test::as_string(3.14) == "3.14");
    EXPECT(test::as_string(std::string("hello")) == "hello");
    EXPECT(test::as_string('x') == "x");
}

TEST_CASE(as_string_bool)
{
    EXPECT(test::as_string(true) == "true");
    EXPECT(test::as_string(false) == "false");
}

TEST_CASE(as_string_nullptr) { EXPECT(test::as_string(nullptr) == "nullptr"); }

TEST_CASE(as_string_vector)
{
    std::vector<int> v = {1, 2, 3};
    EXPECT(test::as_string(v) == "{ 1, 2, 3}");
}

TEST_CASE(as_string_empty_vector)
{
    std::vector<int> v;
    EXPECT(test::as_string(v) == "{ }");
}

TEST_CASE(as_string_single_element_vector)
{
    std::vector<int> v = {42};
    EXPECT(test::as_string(v) == "{ 42}");
}

TEST_CASE(as_string_pair)
{
    auto p = std::make_pair(1, std::string("two"));
    EXPECT(test::as_string(p) == "{1, two}");
}

TEST_CASE(as_string_optional_with_value)
{
    std::optional<int> o = 5;
    EXPECT(test::as_string(o) == "5");
}

TEST_CASE(as_string_optional_empty)
{
    std::optional<int> o;
    EXPECT(test::as_string(o) == "nullopt");
}

TEST_CASE(as_string_pointer)
{
    int x   = 0;
    int* p  = &x;
    auto s  = test::as_string(p);
    auto sn = test::as_string(static_cast<int*>(nullptr));
    // Non-null pointer should produce some output (hex address)
    EXPECT(not s.empty());
    // Null pointer prints as 0
    EXPECT(not sn.empty());
}

// Tests for expression templates / capture
// Note: The expression template system stores references, so we use local
// variables to avoid dangling references when storing expressions.
TEST_CASE(capture_equal)
{
    int a     = 1;
    int b     = 1;
    auto expr = test::capture{}->*a == b;
    EXPECT(expr.value());
}

TEST_CASE(capture_not_equal)
{
    int a     = 1;
    int b     = 2;
    auto expr = test::capture{}->*a != b;
    EXPECT(expr.value());
}

TEST_CASE(capture_less_than)
{
    int a     = 1;
    int b     = 2;
    auto expr = test::capture{}->*a < b;
    EXPECT(expr.value());
}

TEST_CASE(capture_greater_than)
{
    int a     = 3;
    int b     = 2;
    auto expr = test::capture{}->*a > b;
    EXPECT(expr.value());
}

TEST_CASE(capture_less_than_equal)
{
    int a      = 2;
    int b      = 2;
    int c      = 1;
    auto expr1 = test::capture{}->*a <= b;
    auto expr2 = test::capture{}->*c <= b;
    EXPECT(expr1.value());
    EXPECT(expr2.value());
}

TEST_CASE(capture_greater_than_equal)
{
    int a      = 2;
    int b      = 2;
    int c      = 3;
    auto expr1 = test::capture{}->*a >= b;
    auto expr2 = test::capture{}->*c >= b;
    EXPECT(expr1.value());
    EXPECT(expr2.value());
}

TEST_CASE(capture_and_op)
{
    bool t = true;
    bool f = false;
    // Use EXPECT directly since expression templates store references
    // to temporaries that cannot outlive a single statement
    EXPECT(t and t);
    EXPECT(not(t and f));
    EXPECT(not(f and t));
    EXPECT(not(f and f));
}

TEST_CASE(capture_or_op)
{
    bool t = true;
    bool f = false;
    EXPECT(t or t);
    EXPECT(t or f);
    EXPECT(f or t);
    EXPECT(not(f or f));
}

TEST_CASE(capture_not_op)
{
    bool f = false;
    bool t = true;
    // not operator on lhs_expression only stores a reference to the
    // underlying value, so storing in a variable is safe
    auto expr = not(test::capture{}->*f);
    EXPECT(expr.value());
    auto expr2 = not(test::capture{}->*t);
    EXPECT(not expr2.value());
}

TEST_CASE(expression_to_string)
{
    // Use the EXPECT macro's stringification to verify expression printing.
    // The expression template uses references so we verify via as_string on
    // simpler constructs that don't store intermediate temporaries.
    int a    = 42;
    auto lhs = test::capture{}->*a;
    EXPECT(test::as_string(lhs) == "42");
    EXPECT(test::as_string(not lhs) == "not 42");
}

TEST_CASE(lhs_arithmetic_operators)
{
    int a = 3;
    int b = 10;
    int c = 4;
    EXPECT((test::capture{}->*a + 2).value() == 5);
    EXPECT((test::capture{}->*b - 3).value() == 7);
    EXPECT((test::capture{}->*c * 5).value() == 20);
    EXPECT((test::capture{}->*b / 2).value() == 5);
    EXPECT((test::capture{}->*b % 3).value() == 1);
}

TEST_CASE(lhs_bitwise_operators)
{
    int a = 0xFF;
    int b = 0xF0;
    EXPECT((test::capture{}->*a & 0x0F).value() == 0x0F);
    EXPECT((test::capture{}->*b | 0x0F).value() == 0xFF);
    EXPECT((test::capture{}->*a ^ 0x0F).value() == 0xF0);
}

TEST_CASE(chained_comparison)
{
    int a     = 3;
    auto expr = (test::capture{}->*a + 2) == 5;
    EXPECT(expr.value());
}

// Tests for throws
TEST_CASE(throws_any_exception)
{
    EXPECT(test::throws([] { throw std::runtime_error("err"); }));
    EXPECT(not test::throws([] {}));
}

TEST_CASE(throws_specific_exception)
{
    EXPECT(test::throws<std::runtime_error>([] { throw std::runtime_error("some error"); }));
    EXPECT(not test::throws<std::runtime_error>([] {}));
}

TEST_CASE(throws_with_message)
{
    EXPECT(test::throws<std::runtime_error>(
        [] { throw std::runtime_error("specific error message"); }, "specific"));
    EXPECT(not test::throws<std::runtime_error>(
        [] { throw std::runtime_error("specific error message"); }, "not found"));
}

TEST_CASE(throws_wrong_exception_type)
{
    // Throwing logic_error when expecting runtime_error should not be caught
    bool caught = false;
    try
    {
        test::throws<std::runtime_error>([] { throw std::logic_error("wrong type"); });
    }
    catch(const std::logic_error&)
    {
        caught = true;
    }
    EXPECT(caught);
}

// Tests for within_abs
TEST_CASE(within_abs_close_values)
{
    auto result = test::within_abs(1.0, 1.0 + 1e-7, 1e-6);
    EXPECT(result.value());
}

TEST_CASE(within_abs_exact_values)
{
    auto result = test::within_abs(5.0, 5.0);
    EXPECT(result.value());
}

TEST_CASE(within_abs_far_values)
{
    auto result = test::within_abs(1.0, 2.0, 0.5);
    EXPECT(not result.value());
}

TEST_CASE(within_abs_negative)
{
    auto result = test::within_abs(-1.0, -1.0 + 1e-8);
    EXPECT(result.value());
}

// Tests for generic_parse
TEST_CASE(generic_parse_basic)
{
    std::vector<std::string> args = {"--flag", "value1", "value2"};
    auto result = test::generic_parse(args, [](const std::string& s) -> std::vector<std::string> {
        if(s == "--flag")
            return {"--flag", "--flag"};
        return {};
    });
    EXPECT(result.count("--flag") > 0);
    EXPECT(result.at("--flag").size() == 2);
    EXPECT(result.at("--flag")[0] == "value1");
    EXPECT(result.at("--flag")[1] == "value2");
}

TEST_CASE(generic_parse_no_flags)
{
    std::vector<std::string> args = {"arg1", "arg2"};
    auto result                   = test::generic_parse(
        args, [](const std::string&) -> std::vector<std::string> { return {}; });
    EXPECT(result.count("") > 0);
    EXPECT(result.at("").size() == 2);
    EXPECT(result.at("")[0] == "arg1");
    EXPECT(result.at("")[1] == "arg2");
}

TEST_CASE(generic_parse_flag_no_value)
{
    std::vector<std::string> args = {"--verbose"};
    auto result = test::generic_parse(args, [](const std::string& s) -> std::vector<std::string> {
        if(s == "--verbose")
            return {"--verbose", ""};
        return {};
    });
    EXPECT(result.count("--verbose") > 0);
}

TEST_CASE(generic_parse_mixed)
{
    // With single-element keyword return, flag stays set and all subsequent
    // non-keyword args accumulate under it
    std::vector<std::string> args = {"pos1", "--opt", "val1", "val2"};
    auto result = test::generic_parse(args, [](const std::string& s) -> std::vector<std::string> {
        if(s == "--opt")
            return {"--opt"};
        return {};
    });
    EXPECT(result.at("").size() == 1);
    EXPECT(result.at("")[0] == "pos1");
    EXPECT(result.at("--opt").size() == 2);
    EXPECT(result.at("--opt")[0] == "val1");
    EXPECT(result.at("--opt")[1] == "val2");
}

TEST_CASE(generic_parse_flag_resets)
{
    // Two-element keyword return {"flag", ""} sets flag then resets to ""
    std::vector<std::string> args = {"pos1", "--flag", "pos2"};
    auto result = test::generic_parse(args, [](const std::string& s) -> std::vector<std::string> {
        if(s == "--flag")
            return {"--flag", ""};
        return {};
    });
    EXPECT(result.count("--flag") > 0);
    EXPECT(result.at("").size() == 2);
    EXPECT(result.at("")[0] == "pos1");
    EXPECT(result.at("")[1] == "pos2");
}

// Tests for driver::parse and create_command
TEST_CASE(driver_parse_basic)
{
    test::driver d;
    const char* argv[] = {"test_exe", "case1", "case2"};
    auto args          = d.parse(3, argv);
    EXPECT(args.at("__exe__").front() == "test_exe");
    EXPECT(args.at("").size() == 2);
    EXPECT(args.at("")[0] == "case1");
    EXPECT(args.at("")[1] == "case2");
}

TEST_CASE(driver_parse_with_flags)
{
    test::driver d;
    const char* argv[] = {"test_exe", "--quiet", "case1"};
    auto args          = d.parse(3, argv);
    EXPECT(args.count("--quiet") > 0);
    EXPECT(args.at("").size() == 1);
    EXPECT(args.at("")[0] == "case1");
}

TEST_CASE(driver_parse_help_flag)
{
    test::driver d;
    const char* argv[] = {"test_exe", "-h"};
    auto args          = d.parse(2, argv);
    EXPECT(args.count("--help") > 0);
}

TEST_CASE(driver_parse_continue_flag)
{
    test::driver d;
    const char* argv[] = {"test_exe", "-c"};
    auto args          = d.parse(2, argv);
    EXPECT(args.count("--continue") > 0);
}

TEST_CASE(driver_create_command)
{
    test::string_map args;
    args["__exe__"] = {"./bin/test"};
    args[""]        = {"case1"};
    auto cmd        = test::driver::create_command(args);
    EXPECT(cmd.find("./bin/test") != std::string::npos);
    EXPECT(cmd.find("case1") != std::string::npos);
}

// Tests for driver::glob_tests
TEST_CASE(driver_glob_tests_exact)
{
    auto results = test::driver::glob_tests("globbing");
    EXPECT(results.size() == 1);
    EXPECT(results.front().first == "globbing");
}

TEST_CASE(driver_glob_tests_pattern)
{
    auto results = test::driver::glob_tests("as_string_*");
    EXPECT(results.size() > 1);
}

TEST_CASE(driver_glob_tests_no_match)
{
    auto results = test::driver::glob_tests("nonexistent_test_xyz");
    EXPECT(results.empty());
}

// Tests for failures tracking
TEST_CASE(failures_tracking)
{
    test::failures() = 0;
    EXPECT(test::failures().load() == 0);
    test::report_failure();
    EXPECT(test::failures().load() == 1);
    test::report_failure(3);
    EXPECT(test::failures().load() == 4);
    test::failures() = 0;
}

// Tests for skip
TEST_CASE(skip_exception)
{
    EXPECT(test::throws([] { test::skip("reason"); }));
}

TEST_CASE(skip_with_reason)
{
    try
    {
        test::skip("test reason");
    }
    catch(const test::skip_test& s)
    {
        EXPECT(s.reason == "test reason");
        return;
    }
    // Should not reach here
    EXPECT(false);
}

// Tests for failure_error
TEST_CASE(fail_throws)
{
    EXPECT(test::throws([] { test::fail(); }));
}

// Tests for make_predicate
TEST_CASE(make_predicate_true)
{
    auto pred = test::make_predicate("always_true", [] { return true; });
    EXPECT(pred.value());
}

TEST_CASE(make_predicate_false)
{
    auto pred = test::make_predicate("always_false", [] { return false; });
    EXPECT(not pred.value());
}

TEST_CASE(make_predicate_to_string)
{
    auto pred = test::make_predicate("my_check", [] { return true; });
    auto s    = test::as_string(pred);
    EXPECT(s.find("my_check") != std::string::npos);
}

// Tests for make_function
TEST_CASE(make_function_basic)
{
    auto add = test::make_function("add", [](int a, int b) { return a + b == 5; });
    auto r   = add(2, 3);
    EXPECT(r.value());
}

TEST_CASE(make_function_to_string)
{
    auto is_positive = test::make_function("is_positive", [](int a) { return a > 0; });
    auto r           = is_positive(5);
    auto s           = test::as_string(r);
    EXPECT(s.find("is_positive") != std::string::npos);
}

// Tests for CHECK macro (non-fatal)
TEST_CASE(check_macro_passes)
{
    auto saved       = test::failures().load();
    test::failures() = 0;
    CHECK(1 == 1);
    EXPECT(test::failures().load() == 0);
    test::failures() = saved;
}

TEST_CASE(check_macro_failure_increments)
{
    auto saved       = test::failures().load();
    test::failures() = 0;
    CHECK(1 == 2);
    EXPECT(test::failures().load() == 1);
    test::failures() = saved;
}

// Tests for nested containers
TEST_CASE(as_string_nested_vector)
{
    std::vector<std::vector<int>> v = {{1, 2}, {3, 4}};
    auto s                          = test::as_string(v);
    EXPECT(s.find("1") != std::string::npos);
    EXPECT(s.find("4") != std::string::npos);
}

TEST_CASE(as_string_vector_of_strings)
{
    std::vector<std::string> v = {"hello", "world"};
    auto s                     = test::as_string(v);
    EXPECT(s.find("hello") != std::string::npos);
    EXPECT(s.find("world") != std::string::npos);
}

// Tests for operator objects
TEST_CASE(operator_objects_as_string)
{
    EXPECT(test::equal::as_string() == "==");
    EXPECT(test::not_equal::as_string() == "!=");
    EXPECT(test::less_than::as_string() == "<");
    EXPECT(test::greater_than::as_string() == ">");
    EXPECT(test::less_than_equal::as_string() == "<=");
    EXPECT(test::greater_than_equal::as_string() == ">=");
    EXPECT(test::and_op::as_string() == "and");
    EXPECT(test::or_op::as_string() == "or");
    EXPECT(test::not_op::as_string() == "not");
    EXPECT(test::nop::as_string().empty());
}

TEST_CASE(operator_objects_call)
{
    EXPECT(test::equal::call(1, 1));
    EXPECT(not test::equal::call(1, 2));
    EXPECT(test::not_equal::call(1, 2));
    EXPECT(test::less_than::call(1, 2));
    EXPECT(test::greater_than::call(2, 1));
    EXPECT(test::less_than_equal::call(1, 1));
    EXPECT(test::greater_than_equal::call(2, 2));
    EXPECT(test::and_op::call(true, true));
    EXPECT(not test::and_op::call(true, false));
    EXPECT(test::or_op::call(false, true));
    EXPECT(test::not_op::call(false));
}

TEST_CASE(nop_call)
{
    EXPECT(test::nop::call(42) == 42);
    EXPECT(test::nop::call(std::string("test")) == "test");
}

// Tests for expression value propagation
TEST_CASE(expression_false_value)
{
    int a     = 1;
    int b     = 2;
    auto expr = test::capture{}->*a == b;
    EXPECT(not expr.value());
}

TEST_CASE(expression_chained)
{
    int a = 1;
    int b = 1;
    // Binary expressions with `and` store references to temporaries,
    // so they must be used inline via EXPECT
    EXPECT((a == b) and (2 == 2));
    EXPECT(not((a != b) and (2 == 2)));
}

// Edge cases for glob_match
TEST_CASE(globbing_edge_cases)
{
    // Empty strings
    EXPECT(glob_match("", ""));
    EXPECT(not glob_match("a", ""));
    EXPECT(not glob_match("", "a"));
    EXPECT(glob_match("", "*"));
    EXPECT(not glob_match("", "?"));

    // Only wildcards
    EXPECT(glob_match("anything", "*"));
    EXPECT(glob_match("a", "?"));
    EXPECT(not glob_match("ab", "?"));
    EXPECT(glob_match("ab", "??"));

    // Star at beginning/end
    EXPECT(glob_match("hello", "*hello"));
    EXPECT(glob_match("hello", "hello*"));
    EXPECT(glob_match("hello", "*hello*"));
    EXPECT(glob_match("hello_world", "hello*world"));
    EXPECT(not glob_match("hello_world", "hello*xyz"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
