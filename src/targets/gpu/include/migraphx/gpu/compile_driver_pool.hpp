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
#ifndef MIGRAPHX_GUARD_GPU_COMPILE_DRIVER_POOL_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_DRIVER_POOL_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/process.hpp>
#include <migraphx/value.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Idle sessions of `migraphx-hiprtc-driver --serve`. A request takes an idle session, or starts one
// when none is idle, and puts it back after the reply. The mutex guards only the idle list and the
// driver lookup, so requests on different sessions run in parallel.
struct MIGRAPHX_GPU_EXPORT compile_driver_pool
{
    // Uses the driver find_hiprtc_driver finds, looked up on the first request
    compile_driver_pool() = default;
    explicit compile_driver_pool(fs::path driver_path);

    // Sends `req` to a session and returns the reply, or nullopt when there are no sessions to
    // send it to: the driver wasn't found, or a session failed to start twice in a row. Either
    // way every later request returns nullopt too, and the caller does the work itself. Throws if
    // the session fails once the request is out. A session that replies "timeout" has exited, and
    // one whose request throws is broken, so neither goes back to the pool.
    optional<value> request(const value& req);

    // Closes the idle sessions
    void close();

    std::size_t sessions_started() const { return started; }
    std::size_t sessions_dropped() const { return dropped; }

    private:
    optional<process::session> take();
    void put_back(process::session s);
    void give_up(const std::string& reason);

    std::mutex mutex;
    std::vector<process::session> idle;
    optional<fs::path> driver;
    bool driver_looked_up = false;
    std::atomic<bool> unavailable{false};
    std::atomic<std::size_t> started{0};
    std::atomic<std::size_t> dropped{0};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_DRIVER_POOL_HPP
