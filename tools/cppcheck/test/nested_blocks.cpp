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
// Test for NestedBlocks check
// Note: The NestedBlocks checker only detects blocks directly inside
// control statements (if/for/while/switch), not pure nested blocks in function bodies.

void test_unnecessary_nested_blocks_1()
{
    int x = 5;
    // TODO: migraphx-NestedBlocks false negative - pure nested blocks not detected
    {{x = 10;
}
}
(void)x; // Use variable to avoid warning
}

void test_unnecessary_nested_blocks_2()
{
    int y = 10;
    // TODO: migraphx-NestedBlocks false negative - pure nested blocks not detected
    {{{y = 20;
}
}
}
(void)y; // Use variable to avoid warning
}

void test_necessary_scope_blocks()
{
    {
        int x = 5;
        (void)x;
    }
    {
        int x = 10;
        (void)x;
    }
}

void test_if_statement_blocks(int x)
{
    if(x > 0)
    {
        x = 10;
    }
    (void)x; // Use variable to avoid warning
}

void test_loop_blocks()
{
    for(int i = 0; i < 10; i++)
    {
        int temp = i * 2;
        (void)temp; // Use variable to avoid warning
    }
}
