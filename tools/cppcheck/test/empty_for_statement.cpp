// Test for EmptyForStatement check

void test_empty_for_loop()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int i = 0; i < 10; i++) {}
}

void test_empty_for_with_different_condition()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int j = 10; j > 0; j--) {}
}

void test_for_with_statement()
{
    for(int i = 0; i < 10; i++)
    {
        i += 2;
    }
}

void test_for_with_break()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
            break;
    }
}

void test_range_based_for()
{
    const int arr[] = {1, 2, 3, 4, 5};
    for(int x : arr)
    {
        x = x * 2;
        (void)x; // Suppress warning
    }
}
