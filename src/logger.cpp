/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/logger.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace migraphx {
namespace log {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_LEVEL)

std::string get_formatted_timestamp(std::chrono::time_point<std::chrono::system_clock> time)
{
    auto now_in_time_t   = std::chrono::system_clock::to_time_t(time);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()) %
                1000000;
    auto* now_as_tm_date = std::localtime(&now_in_time_t);
    std::stringstream ss;
    ss << std::put_time(now_as_tm_date, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0')
        << std::setw(6) << us.count();
    return ss.str();
}

void record(severity s, std::string_view msg, source_location loc)
{
    // Create log header
    std::string header = get_formatted_timestamp(std::chrono::system_clock::now()) + " [" + "_EWIDT"[
                             static_cast<size_t>(s)] + "] [" + loc.file_name() + ":" +
                         std::to_string(loc.line()) + "] ";
    std::string message = header + std::string(msg) + "\n";
                      
#ifndef _WIN32
    // Add color if output is a terminal
    static const bool use_color = isatty(STDERR_FILENO) != 0;
    if(use_color)
    {
        switch(s)
        {
        case severity::WARN:
            message = "\033[33m" + message + "\033[0m"; // Yellow
            break;
        case severity::ERROR:
            message = "\033[31m" + message + "\033[0m"; // Red
            break;
        case severity::TRACE:
        case severity::DEBUG:
        case severity::INFO:
        case severity::NONE: break; // Prevents -Wswitch-enum warning during compilation
        }
    }
#endif

    std::cerr << message;
}

bool is_enabled(severity s)
{
    static auto level = value_of(MIGRAPHX_LOG_LEVEL{}, static_cast<size_t>(severity::INFO));
    return static_cast<size_t>(s) <= level;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace logger
} // namespace migraphx
