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
// Test for UnnecessaryEmptyCondition check
#include <cstddef>
#include <string>
#include <vector>

void test_positive_cases()
{
    std::vector<int> container = {1, 2, 3, 4, 5};

    // Should trigger: unnecessary empty check before range-based for
    // cppcheck-suppress migraphx-UnnecessaryEmptyCondition
    if(not container.empty())
    {
        for(auto item : container)
        {
            // Process item
            int x = item * 2;
            (void)x;
        }
    }

    // Should trigger: another case with different variable name
    std::vector<std::string> items = {"a", "b", "c"};
    // cppcheck-suppress migraphx-UnnecessaryEmptyCondition
    if(not items.empty())
    {
        for(const auto& item : items)
        {
            // Process item
            std::string processed = item + "_processed";
            (void)processed;
        }
    }
}

void test_negative_cases()
{
    std::vector<int> container = {1, 2, 3, 4, 5};

    // Should not trigger: different containers
    std::vector<int> other_container = {6, 7, 8};
    if(not container.empty())
    {
        for(auto item : other_container)
        {
            int x = item * 2;
            (void)x;
        }
    }

    // Should not trigger: no empty check
    for(auto item : container)
    {
        int x = item * 2;
        (void)x;
    }

    // Should not trigger: traditional for loop
    if(not container.empty())
    {
        for(size_t i = 0; i < container.size(); i++)
        {
            int x = container[i] * 2;
            (void)x;
        }
    }

    // Should not trigger: checking size instead of empty
    if(container.size() > 0)
    {
        for(auto item : container)
        {
            int x = item * 2;
            (void)x;
        }
    }
}
