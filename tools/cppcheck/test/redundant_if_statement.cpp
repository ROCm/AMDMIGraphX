// Test for RedundantIfStatement check

bool test_redundant_if_return_boolean_1()
{
    bool condition = true;
    // cppcheck-suppress migraphx-RedundantIfStatement
    if(condition)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool test_redundant_if_return_boolean_2()
{
    bool condition = false;
    // cppcheck-suppress migraphx-RedundantIfStatement
    if(condition)
    {
        return false;
    }
    else
    {
        return true;
    }
}

bool test_same_return_values(int x)
{
    if(x > 0)
    {
        return true;
    }
    else
    {
        return true;
    }
}

bool test_complex_boolean_returns(int x)
{
    if(x > 0)
    {
        return x > 5;
    }
    else
    {
        return x < 0;
    }
}

int test_non_boolean_returns(int x)
{
    if(x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
