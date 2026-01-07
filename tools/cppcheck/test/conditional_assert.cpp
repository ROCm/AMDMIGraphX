// Test for ConditionalAssert check
#include <cassert>

void test_redundant_if_before_assert(int x)
{
    // cppcheck-suppress migraphx-ConditionalAssert
    if(x > 0)
    {
        assert(x > 0);
    }
}

void test_redundant_if_before_assert_different_condition(int x)
{
    // cppcheck-suppress migraphx-ConditionalAssert
    if(x != 0)
    {
        assert(x != 0);
    }
}

void test_different_conditions(int x)
{
    // cppcheck-suppress migraphx-ConditionalAssert
    if(x > 0)
    {
        assert(x < 10);
    }
}

void test_assert_without_if(int x) { assert(x > 0); }

void test_if_without_assert(int& x)
{
    if(x > 0)
    {
        x = x + 1;
    }
}

void test_multiple_statements_in_if(int x)
{
    if(x > 0)
    {
        int y = x * 2;
        assert(x > 0);
        (void)y; // Use variable to avoid warning
    }
}
