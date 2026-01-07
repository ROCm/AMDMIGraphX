// Test for MultipleUnaryOperator check

void test_double_negative()
{
    int x = 5;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    int y = -(-x);
    (void)y; // Use variable to avoid warning
}

void test_double_positive()
{
    int x = 5;
    // TODO: migraphx-MultipleUnaryOperator false negative - double positive not detected
    int z = +(+x);
    (void)z; // Use variable to avoid warning
}

void test_double_not()
{
    bool b = true;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    bool result = not(not b);
    (void)result; // Use variable to avoid warning
}

void test_multiple_bitwise_not()
{
    unsigned int u = 0xFF;
    // cppcheck-suppress migraphx-MultipleUnaryOperator
    unsigned int inverted = ~~u;
    (void)inverted; // Use variable to avoid warning
}

void test_single_unary_operators()
{
    int x = 5;
    int y = -x;
    int z = +x;
    (void)y; // Use variables to avoid warnings
    (void)z;
}

void test_single_logical_not()
{
    bool b      = true;
    bool result = not b;
    (void)result; // Use variable to avoid warning
}

void test_binary_operators()
{
    int x  = 5;
    int y  = 10;
    int z  = 15;
    int a  = x + y;
    int b2 = x - z;
    (void)a; // Use variables to avoid warnings
    (void)b2;
}

void test_increment_decrement()
{
    int x = 5;
    x++;
    ++x;
    x--;
    --x;
    (void)x;
}
