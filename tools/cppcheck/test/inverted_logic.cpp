// Test for InvertedLogic rule from rules.xml

void test_inverted_logic_with_if_else_1()
{
    int x = 5;
    int y = 10;
    // cppcheck-suppress InvertedLogic
    if(x != y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
}

void test_inverted_logic_with_negation()
{
    int x     = 5;
    bool flag = true;
    // cppcheck-suppress InvertedLogic
    if(!flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
}

void test_inverted_logic_ternary_1()
{
    int x = 5;
    int y = 10;
    // cppcheck-suppress InvertedLogic
    int result1 = (x != y) ? 0 : 1;
}

void test_inverted_logic_ternary_2()
{
    bool flag = true;
    // cppcheck-suppress InvertedLogic
    int result2 = (!flag) ? 0 : 1;
}

void test_positive_logic_equality()
{
    int x = 5;
    int y = 10;
    if(x == y)
    {
        x = 0;
    }
    else
    {
        x = 1;
    }
}

void test_positive_logic_boolean()
{
    int x     = 5;
    bool flag = true;
    if(flag)
    {
        x = 2;
    }
    else
    {
        x = 3;
    }
}

void test_other_comparisons()
{
    int x = 5;
    int y = 10;
    if(x > y)
    {
        x = 4;
    }
    else
    {
        x = 5;
    }
}

void test_positive_ternary()
{
    int x       = 5;
    int y       = 10;
    bool flag   = true;
    int result1 = (x == y) ? 0 : 1;
    int result2 = flag ? 0 : 1;
}
