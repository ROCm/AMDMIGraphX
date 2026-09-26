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
#include <migraphx/par_for.hpp>
#include <migraphx/errors.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include <test.hpp>

TEST_CASE(par_for_runs_all)
{
    std::vector<int> data(64, 0);
    migraphx::par_for(data.size(), 1, [&](std::size_t i) { data[i] = 1; });
    EXPECT(std::all_of(data.begin(), data.end(), [](int x) { return x == 1; }));
}

TEST_CASE(par_for_exception_propagates)
{
    EXPECT(test::throws<migraphx::exception>(
        [] {
            migraphx::par_for(64, 1, [](std::size_t i) {
                if(i % 2 == 0)
                    MIGRAPHX_THROW("par_for_error");
            });
        },
        "par_for_error"));
}

TEST_CASE(par_for_exception_propagates_serial)
{
    EXPECT(test::throws<migraphx::exception>(
        [] {
            migraphx::par_for(4, 100, [](std::size_t i) {
                if(i == 2)
                    MIGRAPHX_THROW("par_for_error");
            });
        },
        "par_for_error"));
}

TEST_CASE(dynamic_par_for_runs_each_once)
{
    std::vector<int> data(64, 0);
    migraphx::dynamic_par_for(data.size(), 4, [&](std::size_t i) { data[i]++; });
    EXPECT(std::all_of(data.begin(), data.end(), [](int x) { return x == 1; }));
}

TEST_CASE(dynamic_par_for_no_items)
{
    std::atomic<bool> called{false};
    migraphx::dynamic_par_for(0, 4, [&](std::size_t) { called = true; });
    EXPECT(not called);
}

TEST_CASE(dynamic_par_for_hands_out_one_index_at_a_time)
{
    // Index 0 waits for all the others, which static chunks would leave queued behind it
    std::mutex m;
    std::condition_variable cv;
    std::size_t done = 0;
    bool waited      = false;
    migraphx::dynamic_par_for(64, 2, [&](std::size_t i) {
        std::unique_lock<std::mutex> lock(m);
        if(i == 0)
        {
            waited = cv.wait_for(lock, std::chrono::seconds{10}, [&] { return done == 63; });
            return;
        }
        ++done;
        cv.notify_all();
    });
    EXPECT(waited);
}

TEST_CASE(dynamic_par_for_one_thread_runs_on_caller)
{
    std::vector<std::thread::id> ids(8);
    migraphx::dynamic_par_for(
        ids.size(), 1, [&](std::size_t i) { ids[i] = std::this_thread::get_id(); });
    EXPECT(std::all_of(
        ids.begin(), ids.end(), [](auto id) { return id == std::this_thread::get_id(); }));
}

TEST_CASE(dynamic_par_for_exception_propagates)
{
    EXPECT(test::throws<migraphx::exception>(
        [] {
            migraphx::dynamic_par_for(64, 4, [](std::size_t i) {
                if(i % 2 == 0)
                    MIGRAPHX_THROW("dynamic_par_for_error");
            });
        },
        "dynamic_par_for_error"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
