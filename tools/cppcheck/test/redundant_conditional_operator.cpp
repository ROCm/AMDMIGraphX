// Test for RedundantConditionalOperator check

void test_redundant_ternary_true_false(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result1 = condition ? true : false;
    (void)result1; // Use variable to avoid warning
}

void test_redundant_ternary_false_true(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result2 = condition ? false : true;
    (void)result2; // Use variable to avoid warning
}

void test_redundant_ternary_both_true(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result3 = condition ? true : true;
    (void)result3; // Use variable to avoid warning
}

void test_redundant_ternary_both_false(bool condition)
{
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result4 = condition ? false : false;
    (void)result4; // Use variable to avoid warning
}

void test_different_values(bool condition, int x, int y)
{
    int result1 = condition ? x : y;
    (void)result1; // Use variable to avoid warning
}

void test_non_boolean_values(bool condition)
{
    int result2 = condition ? 1 : 0;
    (void)result2; // Use variable to avoid warning
}

void test_expressions(bool condition, int x, int y)
{
    int result3 = condition ? x + 1 : y - 1;
    (void)result3; // Use variable to avoid warning
}

void test_function_calls(bool condition, int x)
{
    int result4 = condition ? x : 42;
    (void)result4; // Use variable to avoid warning
}
