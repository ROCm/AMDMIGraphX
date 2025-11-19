// Test for InvertedLogic rule from rules.xml

void test_inverted_logic_with_if_else_1(int x, int y)
{
    // cppcheck-suppress InvertedLogic
    if(x != y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
    (void)x; // Use variable to avoid warning
}

void test_inverted_logic_with_negation(bool flag)
{
    int x = 5;
    // cppcheck-suppress InvertedLogic
    if(!flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
    (void)x; // Use variable to avoid warning
}

void test_inverted_logic_ternary_1(int x, int y)
{
    // cppcheck-suppress InvertedLogic
    int result1 = (x != y) ? 0 : 1;
    (void)result1; // Use variable to avoid warning
}

void test_inverted_logic_ternary_2(bool flag)
{
    // cppcheck-suppress InvertedLogic
    int result2 = (!flag) ? 0 : 1;
    (void)result2; // Use variable to avoid warning
}

void test_positive_logic_equality(int x, int y)
{
    if(x == y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
    (void)x; // Use variable to avoid warning
}

void test_positive_logic_boolean(bool flag)
{
    int x = 5;
    if(flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
    (void)x; // Use variable to avoid warning
}

void test_other_comparisons(int x, int y)
{
    if(x > y)
    {
        x = 4;
    }
    else
    {
        x = 5;
    }
    (void)x; // Use variable to avoid warning
}

void test_positive_ternary(int x, int y, bool flag)
{
    int result1 = (x == y) ? 0 : 1;
    int result2 = flag ? 0 : 1;
    (void)result1; // Use variables to avoid warnings
    (void)result2;
}
