// Test for NestedBlocks check

void test_unnecessary_nested_blocks_1()
{
    int x = 5;
    // TODO: migraphx-NestedBlocks not triggered
    {
        {
            x = 10;
        }
    }
    (void)x; // Use variable to avoid warning
    }

void test_unnecessary_nested_blocks_2()
{
    int y = 10;
    // TODO: migraphx-NestedBlocks not triggered
    {
        {
            {
                y = 20;
            }
        }
    }
    (void)y; // Use variable to avoid warning
    }

void test_necessary_scope_blocks()
{
    {
        int x = 5;
        (void)x;
    }
    {
        int x = 10;
        (void)x;
    }
}

void test_if_statement_blocks(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x; // Use variable to avoid warning
}

void test_loop_blocks()
{
    for(int i = 0; i < 10; i++)
    {
        int temp = i * 2;
        (void)temp; // Use variable to avoid warning
    }
}
