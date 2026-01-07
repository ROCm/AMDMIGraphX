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
// Test for StrlenEmptyString rule from rules.xml
#include <cstring>

void test_strlen_greater_than_zero(const char* str)
{
    // cppcheck-suppress StrlenEmptyString
    if(strlen(str) > 0)
    {
        (void)0; // String is not empty
    }
}

void test_strlen_empty_string_check(const char* str)
{
    // cppcheck-suppress StrlenEmptyString
    if(strlen(str) > 0)
    {
        (void)0; // String is not empty
    }
}

void test_strlen_negated_condition(const char* str)
{
    // cppcheck-suppress StrlenEmptyString
    // cppcheck-suppress UseNamedLogicOperator
    if(!strlen(str))
    {
        (void)0; // String is empty
    }
}

void test_strlen_specific_length_should_not_trigger(const char* str)
{
    // Should not trigger: checking actual length, not emptiness
    if(strlen(str) == 5)
    {
        (void)0; // String has specific length
    }
}

void test_strlen_for_assignment_should_not_trigger(const char* str)
{
    // Should not trigger: using length for other purposes
    int len = strlen(str);
    (void)len; // Suppress unused variable warning
}

void test_direct_empty_check_should_not_trigger(const char* str)
{
    // Should not trigger: direct empty check without strlen
    if(str[0] == '\0')
    {
        (void)0; // String is empty
    }
}

void test_strcmp_should_not_trigger(const char* str)
{
    // Should not trigger: comparing strings, not checking emptiness
    if(strcmp(str, "hello") == 0)
    {
        (void)0; // Strings are equal
    }
}

// Mock strlen and strcmp for compilation
size_t strlen(const char*) { return 0; }
int strcmp(const char*, const char*) { return 0; }
