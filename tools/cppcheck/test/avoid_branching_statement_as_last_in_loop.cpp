// Test for AvoidBranchingStatementAsLastInLoop check

void test_break_as_last_in_for()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop not triggered
    for(int i = 0; i < 10; i++)
    {
        break;
    }
}

void test_continue_as_last_in_while()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop not triggered
    while(true)
    {
        // cppcheck-suppress redundantContinue
        continue;
    }
}

void test_return_as_last_in_for()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop not triggered
    for(int i = 0; i < 10; i++)
    {
        return;
    }
}

void test_break_after_statement()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop not triggered
    for(int i = 0; i < 10; i++)
    {
        int x = 5;
        (void)x;
        break;
    }
}

void test_break_not_last()
{
    for(int i = 0; i < 10; i++)
    {
        break;
        int x = 5; // cppcheck-suppress unreachableCode
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
    for(int i = 0; i < 10; i++) {}
}

void test_break_not_last_complex()
{
    // cppcheck-suppress knownConditionTrueFalse
    while(true)
    {
        int x = 1;
        // cppcheck-suppress knownConditionTrueFalse
        if(x > 0)
            break;
        int y = 2;
        (void)y;
    }
}
