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
#include <migraphx/system.hpp>
#include <algorithm>
#include <iterator>
#include <thread>
#include <vector>

#include <test.hpp>

TEST_CASE(physical_cpu_cores_at_most_logical_cpus)
{
    auto cores = migraphx::physical_cpu_cores();
    EXPECT(cores >= 1);
    EXPECT(cores <= std::max(1u, std::thread::hardware_concurrency()));
}

TEST_CASE(trim_heap_keeps_live_memory)
{
    // Allocated in turn, so freeing one set leaves holes between the blocks of the other
    std::vector<std::vector<std::size_t>> live;
    std::vector<std::vector<std::size_t>> freed;
    std::size_t n = 0;
    std::generate_n(std::back_inserter(live), 64, [&] {
        freed.emplace_back(1024, n);
        return std::vector<std::size_t>(1024, n++);
    });
    freed.clear();
    migraphx::trim_heap();
    std::size_t expected = 0;
    EXPECT(std::all_of(live.begin(), live.end(), [&](const auto& block) {
        auto value = expected++;
        return std::all_of(block.begin(), block.end(), [&](auto x) { return x == value; });
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
