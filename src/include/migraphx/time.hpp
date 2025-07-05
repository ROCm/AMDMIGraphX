/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_RTGLIB_TIME_HPP
#define MIGRAPHX_GUARD_RTGLIB_TIME_HPP

#include <migraphx/config.hpp>
#include <migraphx/source_location.hpp>
#include <chrono>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct timer
{
    using milliseconds = std::chrono::duration<double, std::milli>;
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    template <class Duration>
    auto record() const
    {
        auto finish = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Duration>(finish - start).count();
    }
};

template <class Duration, class F>
auto time(F f)
{
    timer t{};
    f();
    return t.record<Duration>();
}

template <class Duration>
struct auto_timer_log
{
    auto_timer_log(const std::string& pname = "", source_location loc = source_location::current())
        : name(pname), location(loc), t()
    {
        log("Starting timer");
    }

    auto_timer_log(const auto_timer_log&)            = delete;
    auto_timer_log& operator=(const auto_timer_log&) = delete;

    void checkpoint(source_location loc = source_location::current()) const
    {
        log("Checkpoint: ", t.record<Duration>(), "ms", " at line ", loc.line());
    }

    ~auto_timer_log() { log("Finished: ", t.record<Duration>(), "ms"); }

    private:
    std::string name;
    source_location location;
    timer t;

    template <class... Ts>
    void log(const Ts&... xs) const
    {
        std::cout << "[" << location.file_name() << ":" << location.line() << ": "
                  << location.function_name() << ": " << name << "] ";
        (std::cout << ... << xs);
        std::cout << std::endl;
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
