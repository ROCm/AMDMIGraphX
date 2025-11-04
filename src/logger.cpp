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
#include "spdlog/sinks/basic_file_sink.h"

#ifndef _WIN32
#include "spdlog/sinks/stdout_color_sinks.h"
#endif

namespace migraphx {
namespace log {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_LEVEL)

static spdlog::logger* get_migraphx_logger()
{
    static std::vector<spdlog::sink_ptr> sinks;
    static spdlog::logger* migraphx_logger =
        new spdlog::logger("migraphx_logger", begin(sinks), end(sinks));
    return migraphx_logger;
}

void add_file_logger(std::string_view filename)
{
    auto* logger   = get_migraphx_logger();
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string(filename));
    file_sink->set_pattern("%Y-%m-%d %H:%M:%S.%f [%L] [%s:%#] %v");
    logger->sinks().push_back(file_sink);
}

static size_t& get_log_level()
{
    static size_t level = value_of(MIGRAPHX_LOG_LEVEL{}, static_cast<size_t>(severity::INFO));
    return level;
}

static spdlog::level::level_enum to_spdlog_level(severity s)
{
    // Convert migraphx severity to spdlog level
    // migraphx: NONE(0) < ERROR(1) < WARN(2) < INFO(3) < DEBUG(4) < TRACE(5)
    // spdlog:   off(6) > critical(5) > err(4) > warn(3) > info(2) > debug(1) > trace(0)
    switch(s)
    {
    case severity::NONE: return spdlog::level::off;
    case severity::ERROR: return spdlog::level::err;
    case severity::WARN: return spdlog::level::warn;
    case severity::INFO: return spdlog::level::info;
    case severity::DEBUG: return spdlog::level::debug;
    case severity::TRACE: return spdlog::level::trace;
    }
    return spdlog::level::info;
}

static void init_stderr_logger()
{
    static bool initialized = false;
    if(!initialized)
    {
        auto* logger = get_migraphx_logger();
#ifndef _WIN32
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
#else
        auto stderr_sink = std::make_shared<spdlog::sinks::stderr_sink_mt>();
#endif
        // Use spdlog pattern with colors for the color sink
        // %^ ... %$ = wrap entire line with color based on log level
        // %Y-%m-%d %H:%M:%S.%f = timestamp with microseconds
        // [%L] = log level
        // [%s:%#] = source file and line
        // %v = the actual message
        stderr_sink->set_pattern("%^%Y-%m-%d %H:%M:%S.%f [%L] [%s:%#] %v%$");
        logger->sinks().push_back(stderr_sink);
        logger->set_level(to_spdlog_level(static_cast<severity>(get_log_level())));
        initialized = true;
    }
}

std::string get_formatted_timestamp(std::chrono::time_point<std::chrono::system_clock> time)
{
    auto now_in_time_t = std::chrono::system_clock::to_time_t(time);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()) % 1000000;
    auto* now_as_tm_date = std::localtime(&now_in_time_t);
    std::stringstream ss;
    ss << std::put_time(now_as_tm_date, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0')
       << std::setw(6) << us.count();
    return ss.str();
}

void record(severity s, std::string_view msg, source_location loc)
{
    init_stderr_logger();
    auto* logger = get_migraphx_logger();

    // Convert migraphx source_location to spdlog source_loc
    spdlog::source_loc spdlog_loc{loc.file_name(), static_cast<int>(loc.line()), ""};

    // Use spdlog's log method with source location
    // The pattern formatting handles timestamp, level, location, and colors
    logger->log(spdlog_loc, to_spdlog_level(s), "{}", msg);
}

bool is_enabled(severity s) { return static_cast<size_t>(s) <= get_log_level(); }

void set_log_level(severity s)
{
    get_log_level() = static_cast<size_t>(s);
    init_stderr_logger(); // Ensure logger is initialized
    auto* logger = get_migraphx_logger();
    logger->set_level(to_spdlog_level(s));
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace log
} // namespace migraphx
