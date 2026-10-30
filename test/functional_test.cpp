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
#include <migraphx/functional.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <test.hpp>

// Deep enough to overflow a fixed-size stack without stack segmentation
static constexpr std::uint64_t deep_n = 1 << 18;

// The std::thread fallback runs the segment on a new thread, so thread
// affinity is only guaranteed by the native implementations
static bool same_thread_segment()
{
    std::thread::id inner;
    migraphx::run_on_new_stack([&] { inner = std::this_thread::get_id(); });
    return inner == std::this_thread::get_id();
}

TEST_CASE(fix_factorial)
{
    auto result = migraphx::fix<std::uint64_t>([](auto self, std::uint64_t n) -> std::uint64_t {
        if(n == 0)
            return 1;
        return n * self(n - 1);
    })(10);
    EXPECT(result == 3628800);
}

TEST_CASE(fix_tree_recursion)
{
    auto result = migraphx::fix<std::uint64_t>([](auto self, std::uint64_t n) -> std::uint64_t {
        if(n < 2)
            return n;
        return self(n - 1) + self(n - 2);
    })(10);
    EXPECT(result == 55);
}

TEST_CASE(fix_multiple_args)
{
    auto result = migraphx::fix<int>([](auto self, int a, int b) -> int {
        if(b == 0)
            return a;
        return self(b, a % b);
    })(54, 24);
    EXPECT(result == 6);
}

TEST_CASE(fix_void)
{
    std::size_t count = 0;
    migraphx::fix([&](auto self, int n) {
        if(n == 0)
            return;
        count++;
        self(n - 1);
    })(8);
    EXPECT(count == 8);
}

TEST_CASE(fix_move_only_result)
{
    auto result = migraphx::fix<std::unique_ptr<int>>([](auto self, int n) -> std::unique_ptr<int> {
        if(n == 0)
            return std::make_unique<int>(0);
        auto x = self(n - 1);
        return std::make_unique<int>(*x + n);
    })(5);
    EXPECT(*result == 15);
}

TEST_CASE(fix_reference_result)
{
    std::array<int, 4> data{{3, 9, 5, 7}};
    auto& result = migraphx::fix<int&>([&](auto self, std::size_t i) -> int& {
        if(i == 0)
            return data[0];
        auto& prev = self(i - 1);
        return data[i] > prev ? data[i] : prev;
    })(data.size() - 1);
    EXPECT(&result == &data[1]);
    result = 42;
    EXPECT(data[1] == 42);
}

TEST_CASE(fix_const_reference_result)
{
    std::array<int, 3> data{{5, 9, 7}};
    const int& result = migraphx::fix<const int&>([&](auto self, std::size_t i) -> const int& {
        if(i == 0)
            return data[0];
        const int& prev = self(i - 1);
        return data[i] > prev ? data[i] : prev;
    })(data.size() - 1);
    EXPECT(&result == &data[1]);
    EXPECT(result == 9);
}

TEST_CASE(fix_shallow_throw)
{
    EXPECT(test::throws<std::runtime_error>(
        [&] {
            migraphx::fix<std::uint64_t>([](auto self, std::uint64_t n) -> std::uint64_t {
                if(n == 0)
                    throw std::runtime_error{"shallow throw"};
                return self(n - 1) + n;
            })(8);
        },
        "shallow throw"));
}

TEST_CASE(fix_deep_value)
{
    auto result = migraphx::fix<std::uint64_t>([](auto self, std::uint64_t i) -> std::uint64_t {
        if(i == 0)
            return 0;
        // Volatile local read after the call keeps each frame live so the
        // recursion cant be collapsed into a loop
        volatile std::uint64_t pad = i;
        auto x                     = self(i - 1);
        return x + pad;
    })(deep_n);
    EXPECT(result == deep_n * (deep_n + 1) / 2);
}

TEST_CASE(fix_deep_void)
{
    std::uint64_t count = 0;
    migraphx::fix([&](auto self, std::uint64_t i) {
        if(i == 0)
            return;
        volatile std::uint64_t pad = i;
        self(i - 1);
        count += pad / i;
    })(deep_n);
    EXPECT(count == deep_n);
}

TEST_CASE(fix_deep_reference_result)
{
    // The returned reference must survive crossing the stack segments
    std::uint64_t x = 0;
    auto& result = migraphx::fix<std::uint64_t&>([&](auto self, std::uint64_t i) -> std::uint64_t& {
        if(i == 0)
            return x;
        auto& r = self(i - 1);
        r += 1;
        return r;
    })(deep_n);
    EXPECT(&result == &x);
    EXPECT(x == deep_n);
}

TEST_CASE(fix_recursive_mutex)
{
    if(not same_thread_segment())
        return;
    // Every frame holds the recursive_mutex across self, so hopping to a new
    // stack must stay on the same thread for the reacquisition to succeed
    std::recursive_mutex m;
    std::size_t count = 0;
    migraphx::fix([&](auto self, std::size_t i) {
        std::lock_guard<std::recursive_mutex> lock{m};
        count++;
        if(i > 0)
            self(i - 1);
    })(2048);
    EXPECT(count == 2049);
}

TEST_CASE(fix_deep_throw)
{
    EXPECT(test::throws<std::runtime_error>(
        [&] {
            migraphx::fix<std::uint64_t>([](auto self, std::uint64_t i) -> std::uint64_t {
                if(i == 0)
                    throw std::runtime_error{"deep throw"};
                return self(i - 1) + i;
            })(deep_n);
        },
        "deep throw"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
