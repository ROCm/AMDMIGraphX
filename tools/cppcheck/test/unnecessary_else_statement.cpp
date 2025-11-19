// Test for UnnecessaryElseStatement rule from rules.xml

int test_unnecessary_else_after_return(int x)
{
    // TODO: UnnecessaryElseStatement not triggered
    if(x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int test_unnecessary_else_after_throw(int x)
{
    // TODO: UnnecessaryElseStatement not triggered
    if(x < 0)
    {
        throw "error";
    }
    else
    {
        return x;
    }
}

void test_unnecessary_else_after_break(int x)
{
    for(int i = 0; i < 10; i++)
    {
        // TODO: UnnecessaryElseStatement not triggered
        if(i == x)
        {
            break;
        }
        else
        {
            continue;
        }
    }
}

void test_unnecessary_else_after_continue(int& x)
{
    for(int i = 0; i < 10; i++)
    {
        // TODO: UnnecessaryElseStatement not triggered
        if(i == x)
        {
            continue;
        }
        else
        {
            x = i;
        }
    }
}

int test_necessary_else_both_paths_return(int x)
{
    // Should not trigger: both branches have meaningful different logic
    if(x > 0)
    {
        x = x * 2;
        return x;
    }
    else
    {
        x = x + 1;
        return x;
    }
}

void test_necessary_else_no_control_flow(int& x)
{
    // Should not trigger: no return/break/continue/throw
    if(x > 0)
    {
        x = x * 2;
    }
    else
    {
        x = x + 1;
    }
}

void test_necessary_else_if_chain(int x)
{
    // Should not trigger: else if chain
    if(x > 0)
    {
        return;
    }
    else if(x < 0)
    {
        return;
    }
}
