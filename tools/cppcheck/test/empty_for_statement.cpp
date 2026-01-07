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
// Test for EmptyForStatement check

void test_empty_for_loop()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int i = 0; i < 10; i++) {}
}

void test_empty_for_with_different_condition()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int j = 10; j > 0; j--) {}
}

void test_for_with_statement()
{
    for(int i = 0; i < 10; i++)
    {
        i += 2;
    }
}

void test_for_with_break()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
            break;
    }
}

void test_range_based_for()
{
    const int arr[] = {1, 2, 3, 4, 5};
    for(int x : arr)
    {
        x = x * 2;
        (void)x; // Suppress warning
    }
}
