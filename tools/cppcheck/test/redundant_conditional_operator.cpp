// Test for RedundantConditionalOperator check

void test_redundant_ternary_true_false()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result1 = condition ? true : false;
}

void test_redundant_ternary_false_true()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result2 = condition ? false : true;
}

void test_redundant_ternary_both_true()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result3 = condition ? true : true;
}

void test_redundant_ternary_both_false()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantConditionalOperator
    bool result4 = condition ? false : false;
}

void test_different_values()
{
    bool condition = true;
    int x = 5, y = 10;
    int result1 = condition ? x : y;
}

void test_non_boolean_values()
{
    bool condition = true;
    int result2    = condition ? 1 : 0;
}

void test_expressions()
{
    bool condition = true;
    int x = 5, y = 10;
    int result3 = condition ? x + 1 : y - 1;
}

void test_function_calls()
{
    bool condition = true;
    int x          = 5;
    int result4    = condition ? x : 42;
}
