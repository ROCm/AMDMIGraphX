// Test for GotoStatement check

void test_goto_usage()
{
    int x = 5;
    if(x > 0)
    {
        // cppcheck-suppress migraphx-GotoStatement
        goto end;
    }
    x = 10;
end:
    return;
}

void test_goto_in_loop()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
        {
            // cppcheck-suppress migraphx-GotoStatement
            goto loop_end;
        }
    }
loop_end:
    return;
}

void test_no_goto()
{
    int x = 5;
    if(x > 0)
    {
        x = 10;
    }
    return;
}

void test_normal_control_flow()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
        {
            break;
        }
    }
}
