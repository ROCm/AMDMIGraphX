// Test for MutableVariable check

void test_mutable_public_member()
{
    class TestClass
    {
        public:
        // cppcheck-suppress migraphx-MutableVariable
        mutable int x;
    };
}

void test_mutable_flag()
{
    class TestClass
    {
        public:
        // cppcheck-suppress migraphx-MutableVariable
        mutable bool flag;
    };
}

void test_mutable_private_member()
{
    class TestClass
    {
        private:
        // cppcheck-suppress migraphx-MutableVariable
        mutable double value;
    };
}

void test_regular_variables()
{
    int x        = 5;
    bool flag    = true;
    double value = 3.14;
}

void test_const_variables()
{
    const int cx     = 10;
    const bool cflag = false;
}

void test_static_variables() { static int sx = 15; }

void test_good_class_members()
{
    class GoodClass
    {
        public:
        int x;
        bool flag;

        private:
        double value;
        const int constant;

        public:
        GoodClass() : constant(42) {}
    };
}
