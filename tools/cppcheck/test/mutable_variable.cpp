// Test for MutableVariable check

void test_mutable_public_member()
{
    class TestClass
    {
        public:
        // TODO: migraphx-MutableVariable not triggered
        mutable int x;
    };
}

void test_mutable_flag()
{
    class TestClass
    {
        public:
        // TODO: migraphx-MutableVariable not triggered
        mutable bool flag;
    };
}

void test_mutable_private_member()
{
    class TestClass
    {
        private:
        // TODO: migraphx-MutableVariable not triggered
        mutable double value;
        public:
        TestClass() : value(0.0) {}
    };
}

void test_regular_variables()
{
    int x        = 5;
    bool flag    = true;
    double value = 3.14;
    (void)x;
    (void)flag;
    (void)value;
}

void test_const_variables()
{
    const int cx     = 10;
    const bool cflag = false;
    (void)cx;
    (void)cflag;
}

void test_static_variables() { static int sx = 15; (void)sx; }

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
        GoodClass() : x(0), flag(false), value(0.0), constant(42) {}
    };
}
