// Test for MatcherNestedParentheses check
// This check finds deeply nested match:: patterns in matcher() functions.
// Deep nesting affects readability; consider extracting intermediate matchers.

// Mock types to simulate MiGraphX matcher infrastructure
namespace match {
struct matcher_t
{
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
        // cppcheck-suppress migraphx-MatcherNestedParentheses
        return match::name("mul")(
            match::either_arg(0, 1)(match::name("add")(match::args(match::is_constant()))));
    }
};

struct test_five_levels
{
    // Should trigger: 5 levels - name(either_arg(name(args(name(used_once())))))
    auto matcher() const
    {
        // cppcheck-suppress migraphx-MatcherNestedParentheses
        return match::name("mul")(match::either_arg(0, 1)(
            match::name("add")(match::args(match::name("conv")(match::used_once())))));
    }
};

struct test_bind_exception
{
    // Should not trigger: bind(((expr)))) pattern is an exception
    // The bind exception requires bind to be immediately before nested parens
    auto matcher() const { return bind((((match::name("mul"))))); }

    match::matcher_t bind(match::matcher_t m) { return m; }
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
