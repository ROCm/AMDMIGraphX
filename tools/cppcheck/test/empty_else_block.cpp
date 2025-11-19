// Test for EmptyElseBlock check

void test_empty_else_1(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    // cppcheck-suppress migraphx-EmptyElseBlock
    else {}
    (void)x;
}

void test_empty_else_2(int x)
{
    if(x < 0)
    {
        x = -x;
    }
    // cppcheck-suppress migraphx-EmptyElseBlock
    else {}
    (void)x;
}

void test_else_with_statement(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    else
    {
        x = 0;
    }
    (void)x;
}

void test_no_else_block(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x;
}

void test_else_if_chain()
{
    int x = 5;
    if(x > 0)
    {
        x = 10;
    }
    else if(x < 0)
    {
        x = 0;
    }
}
