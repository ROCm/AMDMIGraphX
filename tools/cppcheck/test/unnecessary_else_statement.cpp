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
// Test for UnnecessaryElseStatement rule from rules.xml

int test_unnecessary_else_after_return(int x)
{
    // TODO: UnnecessaryElseStatement false negative
    if(x > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int test_unnecessary_else_after_throw(int x)
{
    // TODO: UnnecessaryElseStatement false negative
    if(x < 0)
    {
        throw "error";
    }
    else
    {
        return x;
    }
}

void test_unnecessary_else_after_break(int x)
{
    for(int i = 0; i < 10; i++)
    {
        // TODO: UnnecessaryElseStatement false negative
        if(i == x)
        {
            break;
        }
        else
        {
            continue;
        }
    }
}

void test_unnecessary_else_after_continue(int& x)
{
    for(int i = 0; i < 10; i++)
    {
        // TODO: UnnecessaryElseStatement false negative
        if(i == x)
        {
            continue;
        }
        else
        {
            x = i;
        }
    }
}

int test_necessary_else_both_paths_return(int x)
{
    // TODO: UnnecessaryElseStatement false positive - multiple statements before return
    // cppcheck-suppress UnnecessaryElseStatement
    if(x > 0)
    {
        x = x * 2;
        return x;
    }
    else
    {
        x = x + 1;
        return x;
    }
}

void test_necessary_else_no_control_flow(int& x)
{
    // Should not trigger: no return/break/continue/throw
    if(x > 0)
    {
        x = x * 2;
    }
    else
    {
        x = x + 1;
    }
}

void test_necessary_else_if_chain(int x)
{
    // Should not trigger: else if chain
    if(x > 0)
    {
        return;
    }
    else if(x < 0)
    {
        return;
    }
}
