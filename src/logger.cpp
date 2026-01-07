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
#include "migraphx/functional.hpp"
#include <migraphx/logger.hpp>
#include <migraphx/color.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#ifdef _WIN32
// cppcheck-suppress [definePrefix, defineUpperCase]
#define localtime_r(time_t, tm) localtime_s(tm, time_t)
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace log {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_LEVEL)

// Internal sink entry: stores sink callback and severity level
struct sink_entry
{
    sink callback;
    severity level;
};

static std::string format_timestamp()
{
    auto now        = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_us =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;
    std::tm now_tm{};
    localtime_r(&now_time_t, &now_tm);
    std::ostringstream ss;
    ss << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(6)
       << now_us.count();
    return ss.str();
}

static std::string to_string(severity s)
{
    switch(s)
    {
    case severity::none: return "NONE";
    case severity::error: return "ERROR";
    case severity::warn: return "WARN";
    case severity::info: return "INFO";
    case severity::debug: return "DEBUG";
    case severity::trace: return "TRACE";
    }
    return "UNKNOWN";
}

static color severity_color(severity s)
{
    switch(s)
    {
    case severity::none: return color::reset;
    case severity::error: return color::fg_red;
    case severity::warn: return color::fg_yellow;
    case severity::info: return color::reset;
    case severity::debug: return color::fg_cyan;
    case severity::trace: return color::fg_white;
    }
    return color::reset;
}

// Create the default stderr sink
static sink make_stderr_sink()
{
    return [](severity s, std::string_view msg, source_location loc) {
        std::cerr << severity_color(s) << format_timestamp() << " [" << to_string(s) << "] ["
                  << loc.file_name() << ":" << loc.line() << "] " << msg << color::reset
                  << std::endl;
    };
}

// Create a file sink
static sink make_file_sink(const std::string& filename)
{
    auto file = std::make_shared<std::ofstream>(filename, std::ios::app);
    if(not file->is_open())
    {
        std::cerr << "Failed to open log file: " << filename << std::endl;
    }
    return [file](severity s, std::string_view msg, source_location loc) {
        if(file->is_open())
        {
            *file << format_timestamp() << " [" << to_string(s) << "] [" << loc.file_name() << ":"
                  << loc.line() << "] " << msg << std::endl;
        }
    };
}

static auto& max_enabled_level()
{
    static std::atomic<severity> max{severity::info};
    return max;
}

static void update_enabled_level(const std::vector<std::optional<sink_entry>>& sinks)
{
    auto it = std::max_element(sinks.begin(), sinks.end(), by(std::less<>{}, [](const auto& entry) -> severity {
        if(entry.has_value())
            return entry->level;
        return severity::none;
    }));
    severity max_level = it == sinks.end() ? severity::none : (*it)->level;
    max_enabled_level().store(max_level);
}

// Thread-safe access to sinks (stderr sink is automatically initialized at index 0)
static void access_sinks(const std::function<void(std::vector<std::optional<sink_entry>>&)>& f)
{
    static std::mutex m;
    static auto sinks = []() {
        // cppcheck-suppress migraphx-RedundantCast
        auto level = static_cast<severity>(
            value_of(MIGRAPHX_LOG_LEVEL{}, static_cast<size_t>(severity::info)));

        // If MIGRAPHX_LOG_LEVEL is set, this will store the value into the atomic when first called
        max_enabled_level().store(level);
        return std::vector<std::optional<sink_entry>>{sink_entry{make_stderr_sink(), level}};
    }();
    std::lock_guard<std::mutex> lock(m);
    f(sinks);
}

size_t add_sink(sink s, severity level)
{
    size_t id = 0;
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        // Find an empty slot or add a new one
        auto it = std::find_if(sinks.begin(), sinks.end(), [](const auto& e) { return not e.has_value(); });
        id = it - sinks.begin();
        if(it == sinks.end())
        {
            sinks.push_back(sink_entry{std::move(s), level});
        }
        else
        {
            *it = sink_entry{std::move(s), level};
        }
        update_enabled_level(sinks);
    });
    return id;
}

void remove_sink(size_t id)
{
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        if(id < sinks.size())
        {
            sinks[id] = std::nullopt;
            update_enabled_level(sinks);
        }
    });
}

void set_severity(severity level, size_t id)
{
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        if(id < sinks.size() and sinks[id].has_value())
        {
            sinks[id]->level = level;
            update_enabled_level(sinks);
        }
    });
}

size_t add_file_logger(std::string_view filename, severity level)
{
    return add_sink(make_file_sink(std::string(filename)), level);
}

void record(severity s, std::string_view msg, source_location loc)
{
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        for(auto& entry : sinks)
        {
            if(entry.has_value() and s <= entry->level)
            {
                entry->callback(s, msg, loc);
            }
        }
    });
}

bool is_enabled(severity level) { return level <= max_enabled_level().load(); }

} // namespace log
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
