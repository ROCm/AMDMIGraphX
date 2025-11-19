// Test for NestedBlocks check

void test_unnecessary_nested_blocks_1()
{
    int x = 5;
    // cppcheck-suppress migraphx-NestedBlocks
    {
        {
            x = 10;
        }
    }
}

void test_unnecessary_nested_blocks_2()
{
    int y = 10;
    // cppcheck-suppress migraphx-NestedBlocks
    {
        {
            {
                y = 20;
            }
        }
    }
}

void test_necessary_scope_blocks()
{
    {
        int x = 5;
    }
    {
        int x = 10;
    }
}

void test_if_statement_blocks()
{
    int x = 5;
    if(x > 0)
    {
        x = 10;
    }
}

void test_loop_blocks()
{
    for(int i = 0; i < 10; i++)
    {
        int temp = i * 2;
    }
}
