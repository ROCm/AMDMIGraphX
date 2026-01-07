/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// Test for ForLoopShouldBeWhileLoop check

void test_for_with_empty_init_and_increment_1()
{
    int x = 5;
    // cppcheck-suppress migraphx-ForLoopShouldBeWhileLoop
    for(; x > 0;)
    {
        x--;
    }
}

void test_for_with_empty_init_and_increment_2()
{
    int x = 5;
    // cppcheck-suppress migraphx-ForLoopShouldBeWhileLoop
    for(; x < 10;)
    {
        x++;
    }
}

void test_standard_for_loop()
{
    for(int i = 0; i < 10; i++)
    {
        int x = i;
        (void)x; // Use variable to avoid warning
    }
}

void test_for_with_init_no_increment()
{
    for(int i = 0; i < 10;)
    {
        i++;
    }
}

void test_for_with_increment_no_init()
{
    int x = 5;
    for(; x < 10; x++)
    {
        int y = x;
        (void)y; // Use variable to avoid warning
    }
}

void test_empty_for_loop()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(; false;) {}
}
