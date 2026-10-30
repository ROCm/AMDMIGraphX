/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/stack_segment.hpp>
#include <array>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>
#include <test.hpp>

// The std::thread fallback runs the segment on a new thread, so thread
// affinity is only guaranteed by the native implementations
static bool same_thread_segment()
{
    std::thread::id inner;
    migraphx::run_on_new_stack([&] { inner = std::this_thread::get_id(); });
    return inner == std::this_thread::get_id();
}

TEST_CASE(run_on_new_stack_basic)
{
    int x = 0;
    migraphx::run_on_new_stack([&] { x = 42; });
    EXPECT(x == 42);
}

TEST_CASE(run_on_new_stack_value_capture)
{
    int result = 0;
    int a      = 20;
    int b      = 22;
    migraphx::run_on_new_stack([&result, a, b] { result = a + b; });
    EXPECT(result == 42);
}

TEST_CASE(run_on_new_stack_sequential)
{
    int x = 0;
    migraphx::run_on_new_stack([&] { x += 1; });
    migraphx::run_on_new_stack([&] { x += 2; });
    EXPECT(x == 3);
}

TEST_CASE(run_on_new_stack_nested)
{
    std::vector<int> order;
    migraphx::run_on_new_stack([&] {
        order.push_back(1);
        migraphx::run_on_new_stack([&] { order.push_back(2); });
        order.push_back(3);
    });
    order.push_back(4);
    std::vector<int> expected{1, 2, 3, 4};
    EXPECT(order == expected);
}

static void consume_stack(std::size_t levels)
{
    // Volatile element access so the frame cant be optimized away
    std::array<char, 1024 * 1024> buffer{};
    volatile char& front = buffer.front();
    volatile char& back  = buffer.back();
    front                = 1;
    back                 = 2;
    if(levels > 0)
        migraphx::run_on_new_stack([&] { consume_stack(levels - 1); });
    EXPECT(front == 1);
    EXPECT(back == 2);
}

TEST_CASE(run_on_new_stack_stack_space)
{
    // Each level holds a 1MB frame across its nested call, so 50 live levels
    // need far more stack than a single fixed-size stack provides
    consume_stack(50);
}

TEST_CASE(run_on_new_stack_recursive_mutex)
{
    if(not same_thread_segment())
        return;
    // A recursive_mutex held by the caller can be reacquired inside the
    // segment since the native implementations stay on the same thread
    std::recursive_mutex m;
    std::lock_guard<std::recursive_mutex> outer{m};
    int x = 0;
    migraphx::run_on_new_stack([&] {
        std::lock_guard<std::recursive_mutex> inner{m};
        x = 1;
    });
    EXPECT(x == 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
