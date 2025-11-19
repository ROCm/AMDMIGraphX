// Test for EmptyWhileStatement check

void test_empty_while()
{
    int i = 0;
    // cppcheck-suppress migraphx-EmptyWhileStatement
    while(i++ < 5) {}
}

void test_empty_while_different_condition()
{
    int j = 10;
    // cppcheck-suppress migraphx-EmptyWhileStatement
    while(j-- > 0) {}
}

void test_while_with_statement()
{
    int i = 0;
    while(i < 5)
    {
        i++;
    }
}

void test_while_with_break()
{
    int i = 0;
    while(true)
    {
        if(i++ > 5)
            break;
    }
}
