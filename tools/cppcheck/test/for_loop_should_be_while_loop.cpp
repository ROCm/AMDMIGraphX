// Test for ForLoopShouldBeWhileLoop check

void test_for_with_empty_init_and_increment_1()
{
    int x = 5;
    // cppcheck-suppress migraphx-ForLoopShouldBeWhileLoop
    for(; x > 0;)
    {
        x--;
    }
}

void test_for_with_empty_init_and_increment_2()
{
    int x = 5;
    // cppcheck-suppress migraphx-ForLoopShouldBeWhileLoop
    for(; x < 10;)
    {
        x++;
    }
}

void test_standard_for_loop()
{
    for(int i = 0; i < 10; i++)
    {
        int x = i;
        (void)x; // Use variable to avoid warning
    }
}

void test_for_with_init_no_increment()
{
    for(int i = 0; i < 10;)
    {
        i++;
    }
}

void test_for_with_increment_no_init()
{
    int x = 5;
    for(; x < 10; x++)
    {
        int y = x;
        (void)y; // Use variable to avoid warning
    }
}

void test_empty_for_loop()
{
    for(; false;) {}
}
