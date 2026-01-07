// Test for AvoidBranchingStatementAsLastInLoop check

void test_break_as_last_in_for()
{
    for(int i = 0; i < 10; i++)
    {
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        break;
    }
}

void test_continue_as_last_in_while()
{
    while(true)
    {
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        // cppcheck-suppress redundantContinue
        continue;
    }
}

void test_return_as_last_in_for()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop false negative - return not detected
    for(int i = 0; i < 10; i++)
    {
        return;
    }
}

void test_break_after_statement()
{
    for(int i = 0; i < 10; i++)
    {
        int x = 5;
        (void)x;
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        break;
    }
}

void test_break_not_last()
{
    for(int i = 0; i < 10; i++)
    {
        break;
        int x = 5;
        (void)x;
    }
}

void test_no_branching_statement()
{
    for(int i = 0; i < 10; i++)
    {
        int x = i * 2;
        (void)x;
    }
}

void test_empty_loop()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int i = 0; i < 10; i++) {}
}

void test_break_not_last_complex()
{
    while(true)
    {
        int x = 1;
        if(x > 0)
            break;
        int y = 2;
        (void)y;
    }
}
