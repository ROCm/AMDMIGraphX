// Test for EmptySwitchStatement check

void test_empty_switch()
{
    int x = 5;
    // cppcheck-suppress migraphx-EmptySwitchStatement
    switch(x)
    {
    }
}

void test_empty_switch_with_different_variable()
{
    int y = 10;
    // cppcheck-suppress migraphx-EmptySwitchStatement
    switch(y)
    {
    }
}

void test_switch_with_cases()
{
    int x = 5;
    switch(x)
    {
    case 1: x = 10; break;
    case 2: x = 20; break;
    default: x = 0; break;
    }
    (void)x; // Use variable to avoid warning
}

void test_switch_with_single_case()
{
    int x = 5;
    switch(x)
    {
    case 1: x = 100; break;
    }
    (void)x;
}
