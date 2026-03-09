/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
// Test for MatcherNestedParentheses check
// This check finds deeply nested match:: patterns in matcher() functions.
// Deep nesting affects readability; consider extracting intermediate matchers.

// Mock types to simulate MiGraphX matcher infrastructure
namespace match {
struct matcher_t
{
    // Allow chaining like match::name("x")(match::args(...))
    template <class... Args>
    matcher_t operator()(Args...) const
    {
        return {};
    }
};
template <class... Args>
matcher_t name(Args...)
{
    return {};
}
template <class... Args>
matcher_t args(Args...)
{
    return {};
}
template <class... Args>
matcher_t arg(int, Args...)
{
    return {};
}
template <class... Args>
matcher_t either_arg(int, int)
{
    return {};
}
matcher_t any() { return {}; }
matcher_t is_constant() { return {}; }
matcher_t used_once() { return {}; }
template <class... Args>
matcher_t any_of(Args...)
{
    return {};
}
template <class... Args>
matcher_t none_of(Args...)
{
    return {};
}
struct matcher_result
{
};
} // namespace match

// Helper that returns a matcher (like conv_const_weights() in simplify_algebra.cpp)
auto lit_broadcast() { return match::any_of(match::is_constant(), match::name("broadcast")); }

struct test_deep_nesting
{
    // Should trigger: 4 levels - name(either_arg(name(args(is_constant()))))
    auto matcher() const
    {
        return match::name("mul")(
            // cppcheck-suppress migraphx-MatcherNestedParentheses
            match::either_arg(0, 1)(match::name("add")(match::args(match::is_constant()))));
    }
};

struct test_five_levels
{
    // Should trigger: 5 levels - name(either_arg(name(args(name(used_once())))))
    auto matcher() const
    {
        return match::name("mul")(match::either_arg(0, 1)(
            // cppcheck-suppress migraphx-MatcherNestedParentheses
            match::name("add")(match::args(match::name("conv")(match::used_once())))));
    }
};

struct test_bind_exception
{
    // Should not trigger: bind(((expr)))) pattern is an exception
    // The bind exception requires bind to be immediately before nested parens
    auto matcher() const { return bind((((match::name("mul"))))); }

    match::matcher_t bind(match::matcher_t m) const { return m; }
};

struct test_shallow_nesting
{
    // Should not trigger: only 2 levels - name(args())
    auto matcher() const
    {
        return match::name("add")(match::args(match::any(), match::is_constant()));
    }
};

struct test_three_levels
{
    // Should not trigger: only 3 consecutive closing parens - name(args(name()))
    auto matcher() const { return match::name("dot")(match::args(match::name("slice"))); }
};

struct test_not_matcher_function
{
    // Should not trigger: not a matcher() function
    auto find_pattern() const
    {
        return match::name("mul")(
            match::either_arg(0, 1)(match::name("add")(match::args(match::is_constant()))));
    }
};

struct test_non_const_matcher
{
    // Should not trigger: matcher must be const
    auto matcher()
    {
        return match::name("mul")(
            match::either_arg(0, 1)(match::name("add")(match::args(match::is_constant()))));
    }
};
