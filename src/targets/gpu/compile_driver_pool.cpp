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
#include <migraphx/msgpack.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

value compile_driver_pool::request(const fs::path& driver, const value& req)
{
    auto session = take(driver);
    value reply;
    try
    {
        reply = from_msgpack(
            session.request([&](const process::writer& writer) { to_msgpack(req, writer); }));
    }
    catch(...)
    {
        dropped++;
        throw;
    }
    if(reply.contains("timeout"))
        dropped++;
    else
        put_back(std::move(session));
    return reply;
}

void compile_driver_pool::close()
{
    std::vector<process::session> sessions;
    {
        std::lock_guard<std::mutex> lock(mutex);
        sessions.swap(idle);
    }
    // Destroying a session waits for its driver to exit, which takes milliseconds, so the sessions
    // are destroyed together and outside the lock
    par_for(sessions.size(), 1, [&](std::size_t i) { auto session = std::move(sessions[i]); });
}

process::session compile_driver_pool::take(const fs::path& driver)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(not idle.empty())
        {
            auto session = std::move(idle.back());
            idle.pop_back();
            return session;
        }
    }
    auto session = process{driver, {"--serve"}}.start();
    started++;
    return session;
}

void compile_driver_pool::put_back(process::session s)
{
    std::lock_guard<std::mutex> lock(mutex);
    idle.push_back(std::move(s));
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
