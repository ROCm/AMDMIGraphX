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
// Test for AvoidBranchingStatementAsLastInLoop check

void test_break_as_last_in_for()
{
    for(int i = 0; i < 10; i++)
    {
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        break;
    }
}

void test_continue_as_last_in_while()
{
    while(true)
    {
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        // cppcheck-suppress redundantContinue
        continue;
    }
}

void test_return_as_last_in_for()
{
    // TODO: migraphx-AvoidBranchingStatementAsLastInLoop false negative - return not detected
    for(int i = 0; i < 10; i++)
    {
        return;
    }
}

void test_break_after_statement()
{
    for(int i = 0; i < 10; i++)
    {
        int x = 5;
        (void)x;
        // cppcheck-suppress migraphx-AvoidBranchingStatementAsLastInLoop
        break;
    }
}

void test_break_not_last()
{
    for(int i = 0; i < 10; i++)
    {
        break;
        int x = 5;
        (void)x;
    }
}

void test_no_branching_statement()
{
    for(int i = 0; i < 10; i++)
    {
        int x = i * 2;
        (void)x;
    }
}

void test_empty_loop()
{
    // cppcheck-suppress migraphx-EmptyForStatement
    for(int i = 0; i < 10; i++) {}
}

void test_break_not_last_complex()
{
    while(true)
    {
        int x = 1;
        if(x > 0)
            break;
        int y = 2;
        (void)y;
    }
}
