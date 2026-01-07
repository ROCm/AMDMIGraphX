// Test for EmptyIfStatement check

void test_empty_if_1(int x)
{
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x > 0) {}
}

void test_empty_if_2(int x)
{
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x == 5) {}
}

void test_if_with_statement(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x;
}

void test_if_with_else(int x)
{
    // Empty if body still triggers even with else clause
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x > 0) {}
    else
    {
        x = 0;
    }
    (void)x;
}

void test_if_with_multiple_statements(int x)
{
    if(x > 0)
    {
        int y = x;
        y     = y * 2;
        (void)y;
    }
}
