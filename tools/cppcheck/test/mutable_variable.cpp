/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
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

void test_static_variables()
{
    static int sx = 15;
    (void)sx;
}

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
