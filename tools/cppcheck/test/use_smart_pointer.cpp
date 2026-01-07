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
#include <string>

// Test for UseSmartPointer check

void test_positive_cases()
{
    // Should trigger: new usage without smart pointer
    // cppcheck-suppress migraphx-UseSmartPointer
    int* ptr1 = new int(5);

    // Should trigger: new array
    // cppcheck-suppress migraphx-UseSmartPointer
    int* ptr2 = new int[10];

    // Should trigger: new with class
    // cppcheck-suppress migraphx-UseSmartPointer
    std::string* str_ptr = new std::string("hello");

    // Should trigger: new with user-defined type
    class MyClass
    {
    };
    // cppcheck-suppress migraphx-UseSmartPointer
    MyClass* obj_ptr = new MyClass();

    // Clean up to avoid memory leaks
    delete ptr1;
    delete[] ptr2;
    delete str_ptr;
    delete obj_ptr;
}

void test_negative_cases()
{
    // Should not trigger: using smart pointers
    // std::unique_ptr<int> smart_ptr = std::make_unique<int>(5);
    // std::shared_ptr<int> shared_ptr = std::make_shared<int>(10);

    // Should not trigger: stack allocation
    int local_var = 5;
    int array[10];
    (void)local_var; // Use variables to avoid warnings
    (void)array;

    // Should not trigger: function parameters
    // void func(int* ptr);

    // Should not trigger: member variables
    class TestClass
    {
        int* member_ptr; // This might be managed elsewhere
    };

    // Should not trigger: placement new (advanced case)
    // char buffer[sizeof(int)];
    // int* ptr = new(buffer) int(42);
}
