// Test for EmptyIfStatement check

void test_empty_if_1()
{
    int x = 5;
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x > 0) {}
}

void test_empty_if_2()
{
    int x = 5;
    // cppcheck-suppress migraphx-EmptyIfStatement
    if(x == 5) {}
}

void test_if_with_statement()
{
    int x = 5;
    if(x > 0)
    {
        x = 10;
    }
}

void test_if_with_else()
{
    int x = 5;
    if(x > 0) {}
    else
    {
        x = 0;
    }
}

void test_if_with_multiple_statements()
{
    int x = 5;
    if(x > 0)
    {
        int y = x;
        y     = y * 2;
    }
}
