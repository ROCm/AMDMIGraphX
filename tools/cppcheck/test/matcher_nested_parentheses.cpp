// Test for MatcherNestedParentheses check

class Matcher
{
    public:
    void test_positive_cases()
    {
        // Should trigger: too many nested parentheses in matcher
        // cppcheck-suppress migraphx-MatcherNestedParentheses
        matcher() const { return something((((())))); }

        // Should trigger: another case with deeply nested parentheses
        // cppcheck-suppress migraphx-MatcherNestedParentheses
        matcher() const { return other_func((((some_call())))); }
    }

    void test_negative_cases()
    {
        // Should not trigger: bind function (exception)
        matcher() const { return bind(((()))); }

        // Should not trigger: fewer nested parentheses
        matcher() const { return simple_call(()); }

        // Should not trigger: regular function
        void regular_function() const { return something((((())))); }

        // Should not trigger: not in matcher function
        void other_function() const { return func((((())))); }
    }

    private:
    int something(int) { return 0; }
    int other_func(int) { return 0; }
    int bind(int) { return 0; }
    int simple_call(int) { return 0; }
    int func(int) { return 0; }
};
