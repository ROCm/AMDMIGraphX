/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#include <migraphx/kernels/test.hpp>

using migraphx::test::capture;

// Test capture with temporaries using == operator
TEST_CASE(capture_equal)
{
    auto expr = capture{}->*1 == 1;
    EXPECT(expr.value());
}

// Test capture with temporaries using != operator
TEST_CASE(capture_not_equal)
{
    auto expr = capture{}->*1 != 2;
    EXPECT(expr.value());
}

// Test capture with temporaries using < operator
TEST_CASE(capture_less_than)
{
    auto expr = capture{}->*1 < 2;
    EXPECT(expr.value());
}

// Test capture with temporaries using > operator
TEST_CASE(capture_greater_than)
{
    auto expr = capture{}->*2 > 1;
    EXPECT(expr.value());
}

// Test capture with temporaries using <= operator
TEST_CASE(capture_less_than_equal)
{
    auto expr = capture{}->*1 <= 1;
    EXPECT(expr.value());
}

// Test capture with temporaries using >= operator
TEST_CASE(capture_greater_than_equal)
{
    auto expr = capture{}->*2 >= 2;
    EXPECT(expr.value());
}

// Test capture with and operator
TEST_CASE(capture_and)
{
    auto expr = capture{}->*true and true;
    EXPECT(expr.value());
}

// Test capture with or operator
TEST_CASE(capture_or)
{
    auto expr = capture{}->*false or true;
    EXPECT(expr.value());
}

// Test capture with not operator
TEST_CASE(capture_not)
{
    auto expr = not(capture{}->*false);
    EXPECT(expr.value());
}

// Test expression false value
TEST_CASE(expression_false_value)
{
    auto expr = capture{}->*1 == 2;
    EXPECT(not expr.value());
}

// Test lhs arithmetic: addition
TEST_CASE(lhs_arithmetic_add) { EXPECT((capture{}->*3 + 2).value() == 5); }

// Test lhs arithmetic: subtraction
TEST_CASE(lhs_arithmetic_sub) { EXPECT((capture{}->*5 - 3).value() == 2); }

// Test lhs arithmetic: multiplication
TEST_CASE(lhs_arithmetic_mul) { EXPECT((capture{}->*3 * 4).value() == 12); }

// Test lhs arithmetic: division
TEST_CASE(lhs_arithmetic_div) { EXPECT((capture{}->*12 / 4).value() == 3); }

// Test lhs arithmetic: modulo
TEST_CASE(lhs_arithmetic_mod) { EXPECT((capture{}->*7 % 3).value() == 1); }

// Test lhs bitwise: and
TEST_CASE(lhs_bitwise_and) { EXPECT((capture{}->*0xF & 0x3).value() == 0x3); }

// Test lhs bitwise: or
TEST_CASE(lhs_bitwise_or) { EXPECT((capture{}->*0x1 | 0x2).value() == 0x3); }

// Test lhs bitwise: xor
TEST_CASE(lhs_bitwise_xor) { EXPECT((capture{}->*0xF ^ 0x3).value() == 0xC); }

// Test chained comparison
TEST_CASE(chained_comparison)
{
    auto expr = (capture{}->*3 + 2) == 5;
    EXPECT(expr.value());
}

// Test chained comparison false
TEST_CASE(chained_comparison_false)
{
    auto expr = (capture{}->*3 + 2) == 6;
    EXPECT(not expr.value());
}

// Test operator objects: equal as_string
TEST_CASE(operator_equal_as_string)
{
    const char* s = migraphx::test::equal::as_string();
    EXPECT(s[0] == '=');
    EXPECT(s[1] == '=');
}

// Test operator objects: equal call
TEST_CASE(operator_equal_call)
{
    EXPECT(migraphx::test::equal::call(1, 1));
    EXPECT(not migraphx::test::equal::call(1, 2));
}

// Test operator objects: not_equal call
TEST_CASE(operator_not_equal_call)
{
    EXPECT(migraphx::test::not_equal::call(1, 2));
    EXPECT(not migraphx::test::not_equal::call(1, 1));
}

// Test operator objects: less_than call
TEST_CASE(operator_less_than_call)
{
    EXPECT(migraphx::test::less_than::call(1, 2));
    EXPECT(not migraphx::test::less_than::call(2, 1));
}

// Test operator objects: greater_than call
TEST_CASE(operator_greater_than_call)
{
    EXPECT(migraphx::test::greater_than::call(2, 1));
    EXPECT(not migraphx::test::greater_than::call(1, 2));
}

// Test operator objects: less_than_equal call
TEST_CASE(operator_less_than_equal_call)
{
    EXPECT(migraphx::test::less_than_equal::call(1, 1));
    EXPECT(migraphx::test::less_than_equal::call(1, 2));
    EXPECT(not migraphx::test::less_than_equal::call(2, 1));
}

// Test operator objects: greater_than_equal call
TEST_CASE(operator_greater_than_equal_call)
{
    EXPECT(migraphx::test::greater_than_equal::call(1, 1));
    EXPECT(migraphx::test::greater_than_equal::call(2, 1));
    EXPECT(not migraphx::test::greater_than_equal::call(1, 2));
}

// Test nop call
TEST_CASE(nop_call) { EXPECT(migraphx::test::nop::call(42) == 42); }

// Test nop as_string is empty
TEST_CASE(nop_as_string_empty)
{
    const char* s = migraphx::test::nop::as_string();
    EXPECT(s[0] == '\0');
}

// Test CHECK macro passes without incrementing failures
TEST_CASE(check_macro_passes)
{
    // cppcheck-suppress knownConditionTrueFalse
    CHECK(1 == 1);
}

// Test capture with variables (not just temporaries)
TEST_CASE(capture_with_variables)
{
    int x     = 10;
    int y     = 20;
    auto expr = capture{}->*x < y;
    EXPECT(expr.value());
}

// Test expression chaining with multiple operators
TEST_CASE(expression_chained)
{
    auto expr = (capture{}->*2 + 3) == 5;
    EXPECT(expr.value());
    auto expr2 = (capture{}->*10 - 3) == 7;
    EXPECT(expr2.value());
}

// Test capture preserves value for non-trivial arithmetic
TEST_CASE(capture_complex_arithmetic) { EXPECT((capture{}->*100 / 10 % 3).value() == 1); }

// Test and_op operator object
TEST_CASE(operator_and_op_call)
{
    EXPECT(migraphx::test::and_op::call(true, true));
    EXPECT(not migraphx::test::and_op::call(true, false));
    EXPECT(not migraphx::test::and_op::call(false, true));
    EXPECT(not migraphx::test::and_op::call(false, false));
}

// Test or_op operator object
TEST_CASE(operator_or_op_call)
{
    EXPECT(migraphx::test::or_op::call(true, true));
    EXPECT(migraphx::test::or_op::call(true, false));
    EXPECT(migraphx::test::or_op::call(false, true));
    EXPECT(not migraphx::test::or_op::call(false, false));
}

// Test not_op operator object
TEST_CASE(operator_not_op_call)
{
    EXPECT(migraphx::test::not_op::call(false));
    EXPECT(not migraphx::test::not_op::call(true));
}

// Test function operator object
TEST_CASE(function_call)
{
    auto f = [] { return 42; };
    EXPECT(migraphx::test::function::call(f) == 42);
}

// Test capture with negative values
TEST_CASE(capture_negative_values)
{
    auto expr = capture{}->*(-5) < 0;
    EXPECT(expr.value());
}

// Test capture equality with zero
TEST_CASE(capture_zero)
{
    auto expr = capture{}->*0 == 0;
    EXPECT(expr.value());
}

// Test multiple expressions in sequence
TEST_CASE(multiple_expressions)
{
    auto e1 = capture{}->*1 == 1;
    auto e2 = capture{}->*2 == 2;
    auto e3 = capture{}->*3 == 3;
    EXPECT(e1.value());
    EXPECT(e2.value());
    EXPECT(e3.value());
}

// Test noncopyable type with capture
struct noncopyable
{
    int value;
    constexpr noncopyable(int v) : value(v) {}
    noncopyable(const noncopyable&)            = delete;
    noncopyable& operator=(const noncopyable&) = delete;
    noncopyable(noncopyable&&)                 = default;
    noncopyable& operator=(noncopyable&&)      = default;
    friend constexpr bool operator==(const noncopyable& a, const noncopyable& b)
    {
        return a.value == b.value;
    }
    friend constexpr bool operator!=(const noncopyable& a, const noncopyable& b)
    {
        return a.value != b.value;
    }
};

TEST_CASE(capture_noncopyable)
{
    auto expr = capture{}->*noncopyable{42} == noncopyable{42};
    EXPECT(expr.value());
}

// Test move-only type with capture
struct move_only
{
    int value;
    constexpr move_only(int v) : value(v) {}
    move_only(const move_only&)            = delete;
    move_only& operator=(const move_only&) = delete;
    move_only(move_only&& other)           = default;
    move_only& operator=(move_only&&)      = default;
    friend constexpr bool operator==(const move_only& a, const move_only& b)
    {
        return a.value == b.value;
    }
    friend constexpr bool operator!=(const move_only& a, const move_only& b)
    {
        return a.value != b.value;
    }
};

TEST_CASE(capture_move_only)
{
    auto expr = capture{}->*move_only{42} == move_only{42};
    EXPECT(expr.value());
}
