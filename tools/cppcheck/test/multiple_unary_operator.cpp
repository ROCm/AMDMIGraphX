// Test for MultipleUnaryOperator check

void test_double_negative()
{
    int x = 5;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    int y = -(-x);
}

void test_double_positive()
{
    int x = 5;
    // TODO migraphx-MultipleUnaryOperator
    int z = +(+x);
}

void test_double_not()
{
    bool b = true;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    bool result = not(not b);
}

void test_multiple_bitwise_not()
{
    unsigned int u = 0xFF;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    unsigned int inverted = ~~u;
}

void test_single_unary_operators()
{
    int x = 5;
    int y = -x;
    int z = +x;
}

void test_single_logical_not()
{
    bool b      = true;
    bool result = not b;
}

void test_binary_operators()
{
    int x  = 5;
    int y  = 10;
    int z  = 15;
    int a  = x + y;
    int b2 = x - z;
}

void test_increment_decrement()
{
    int x = 5;
    x++;
    ++x;
    x--;
    --x;
}
