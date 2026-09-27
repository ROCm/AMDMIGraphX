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
#include <migraphx/gpu/compile_driver_pool.hpp>
#include <migraphx/par_for.hpp>
#include <test.hpp>
#include <atomic>

// Once a driver fails to start twice, the pool offers no sessions and doesn't try it again
TEST_CASE(pool_without_driver)
{
    migraphx::gpu::compile_driver_pool pool{"/nonexistent/migraphx-hiprtc-driver"};
    EXPECT(not pool.request({{"not", "a compile request"}}).has_value());
    EXPECT(not pool.request({{"not", "a compile request"}}).has_value());
    EXPECT(pool.sessions_started() == 0);
    EXPECT(pool.sessions_dropped() == 0);
}

// A request the driver can't compile gets an error reply, and the session stays usable
TEST_CASE(pool_reuses_session_after_error_reply)
{
    migraphx::gpu::compile_driver_pool pool;
    auto first = pool.request({{"not", "a compile request"}});
    EXPECT(first.has_value() and first->contains("error"));
    auto second = pool.request({{"not", "a compile request"}});
    EXPECT(second.has_value() and second->contains("error"));
    EXPECT(pool.sessions_started() == 1);
    EXPECT(pool.sessions_dropped() == 0);
}

// Requests that run at once start sessions of their own, which all go back to the pool, so the
// next request starts none until close() stops them
TEST_CASE(pool_concurrent_requests)
{
    migraphx::gpu::compile_driver_pool pool;
    std::atomic<std::size_t> errors{0};
    migraphx::par_for(8, 1, [&](std::size_t) {
        auto reply = pool.request({{"not", "a compile request"}});
        if(reply.has_value() and reply->contains("error"))
            errors++;
    });
    EXPECT(errors == 8);
    auto started = pool.sessions_started();
    EXPECT(started >= 1);
    EXPECT(started <= 8);
    EXPECT(pool.sessions_dropped() == 0);

    pool.request({{"not", "a compile request"}});
    EXPECT(pool.sessions_started() == started);

    pool.close();
    pool.request({{"not", "a compile request"}});
    EXPECT(pool.sessions_started() == started + 1);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
