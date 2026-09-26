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
#include <migraphx/time.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <test.hpp>

// The empty loops below spin to burn CPU time. They stop once the CPU clock reaches its target, so
// a loaded machine only makes them slower. The wall-clock limit only stops a broken clock from
// hanging the test.

TEST_CASE(cpu_timer_counts_busy_work)
{
    migraphx::timer wall{};
    migraphx::cpu_timer cpu{};
    // cppcheck-suppress migraphx-EmptyWhileStatement
    while(cpu.record<std::chrono::milliseconds>() < 50 and wall.record<std::chrono::seconds>() < 10)
    {
    }
    EXPECT(cpu.record<std::chrono::milliseconds>() >= 50);
}

TEST_CASE(cpu_timer_ignores_sleep)
{
    migraphx::timer wall{};
    migraphx::cpu_timer cpu{};
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    EXPECT(wall.record<std::chrono::milliseconds>() >= 200);
    EXPECT(cpu.record<std::chrono::milliseconds>() < 100);
}

TEST_CASE(cpu_timer_counts_other_threads)
{
    migraphx::timer wall{};
    migraphx::cpu_timer cpu{};
    std::atomic<bool> stop{false};
    std::thread spinner{[&] {
        // cppcheck-suppress migraphx-EmptyWhileStatement
        while(not stop)
        {
        }
    }};
    while(cpu.record<std::chrono::milliseconds>() < 50 and wall.record<std::chrono::seconds>() < 10)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    stop = true;
    spinner.join();
    EXPECT(cpu.record<std::chrono::milliseconds>() >= 50);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
