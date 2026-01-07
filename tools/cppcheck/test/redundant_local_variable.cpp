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
// Test for RedundantLocalVariable check

int test_positive_cases()
{
    int x = 5;

    // Should trigger: variable returned immediately after declaration
    int result = x * 2;
    // cppcheck-suppress migraphx-RedundantLocalVariable
    return result;
}

int test_positive_case2(int a, int b)
{
    // Should trigger: complex expression assigned and returned
    int value = a + b * 2;
    // cppcheck-suppress migraphx-RedundantLocalVariable
    return value;
}

int test_negative_cases(int x)
{
    // Should not trigger: variable used before return
    int result = x * 2;
    result     = result + 1;
    return result;
}

int test_negative_case2(int x)
{
    // Should not trigger: multiple statements between declaration and return
    int result = x * 2;
    int temp   = result + 1;
    (void)temp; // Use variable to avoid warning
    return result;
}

int test_negative_case3(int x)
{
    // Should not trigger: different variable returned
    int result = x * 2;
    result     = result + 1;
    return 0;
}

void test_negative_case4(int x)
{
    // Should not trigger: void function
    int result = x * 2;
    (void)result; // Use variable to avoid warning
    return;
}
