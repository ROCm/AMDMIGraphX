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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/par_for.hpp>
#include <exception>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

compile_driver_pool::compile_driver_pool(fs::path driver_path)
    : driver(std::move(driver_path)), driver_looked_up(true)
{
}

optional<value> compile_driver_pool::request(const value& req)
{
    auto session = take();
    if(not session.has_value())
        return nullopt;
    value reply;
    try
    {
        reply = from_msgpack(
            session->request([&](const process::writer& writer) { to_msgpack(req, writer); }));
    }
    catch(...)
    {
        dropped++;
        throw;
    }
    if(reply.contains("timeout"))
        dropped++;
    else
        put_back(std::move(*session));
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

optional<process::session> compile_driver_pool::take()
{
    fs::path path;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(not idle.empty())
        {
            auto session = std::move(idle.back());
            idle.pop_back();
            return session;
        }
        if(not driver_looked_up)
        {
            driver           = find_hiprtc_driver();
            driver_looked_up = true;
        }
        if(driver.has_value())
            path = *driver;
    }
    if(unavailable)
        return nullopt;
    if(path.empty())
    {
        give_up("migraphx-hiprtc-driver was not found");
        return nullopt;
    }
    std::string error;
    auto start = [&]() -> optional<process::session> {
        try
        {
            auto session = process{path, {"--serve"}}.start();
            started++;
            return session;
        }
        catch(const std::exception& e)
        {
            error = e.what();
            return nullopt;
        }
    };
    // A second failure tells a driver that can't serve apart from one session that failed
    auto session = start();
    if(not session.has_value())
        session = start();
    if(not session.has_value())
        give_up("a migraphx-hiprtc-driver session failed to start: " + error);
    return session;
}

void compile_driver_pool::put_back(process::session s)
{
    std::lock_guard<std::mutex> lock(mutex);
    idle.push_back(std::move(s));
}

void compile_driver_pool::give_up(const std::string& reason)
{
    if(not unavailable.exchange(true))
        log::warn() << "MLIR tuning candidates compile without a CPU budget, since " << reason;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
