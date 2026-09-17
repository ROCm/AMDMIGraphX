/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 */
#include <migraphx/migraphx.h>
#include <stdlib.h>
#include <string.h>

void expect_status(migraphx_status x, migraphx_status y)
{
    if(x != y)
        abort();
}

void expect_equal(const char* x, const char* y)
{
    if(strcmp(x, y) != 0)
        abort();
}

int main(void)
{
    char name[1024];
    char truncated_name[2];
    migraphx_operation_t op;
    expect_status(migraphx_operation_create(&op, "add", 0), migraphx_status_success);

    expect_status(migraphx_operation_name(NULL, 1024, op), migraphx_status_bad_param);
    expect_status(migraphx_operation_name(name, 0, op), migraphx_status_bad_param);
    expect_status(migraphx_operation_name(truncated_name, 2, op), migraphx_status_success);
    expect_equal(truncated_name, "a");

    expect_status(migraphx_operation_name(name, 1024, op), migraphx_status_success);
    expect_status(migraphx_operation_destroy(op), migraphx_status_success);
    expect_equal(name, "add");
}
