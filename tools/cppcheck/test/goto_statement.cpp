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
// Test for GotoStatement check

void test_goto_usage(int x)
{
    if(x > 0)
    {
        // cppcheck-suppress migraphx-GotoStatement
        goto end;
    }
    x = 10;
end:
    (void)x; // Use variable to avoid warning
    return;
}

void test_goto_in_loop()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
        {
            // cppcheck-suppress migraphx-GotoStatement
            goto loop_end;
        }
    }
loop_end:
    return;
}

void test_no_goto(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x; // Use variable to avoid warning
    return;
}

void test_normal_control_flow()
{
    for(int i = 0; i < 10; i++)
    {
        if(i == 5)
        {
            break;
        }
    }
}
