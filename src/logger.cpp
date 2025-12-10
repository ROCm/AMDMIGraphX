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
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

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

static char severity_char(severity s)
{
    switch(s)
    {
    case severity::none: return 'N';
    case severity::error: return 'E';
    case severity::warn: return 'W';
    case severity::info: return 'I';
    case severity::debug: return 'D';
    case severity::trace: return 'T';
    }
    return '?';
}

static const char* severity_color(severity s)
{
    switch(s)
    {
    case severity::none: return "";
    case severity::error: return "\033[31m"; // Red
    case severity::warn: return "\033[33m";  // Yellow
    case severity::info: return "";          // Default
    case severity::debug: return "\033[36m"; // Cyan
    case severity::trace: return "\033[37m"; // White/Gray
    }
    return "";
}

static const char* reset_color() { return "\033[0m"; }

// Create the default stderr sink
static sink make_stderr_sink()
{
    return [](severity s, std::string_view msg, source_location loc) {
        std::cerr << severity_color(s) << format_timestamp() << " [" << severity_char(s) << "] ["
                  << loc.file_name() << ":" << loc.line() << "] " << msg << reset_color()
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
            *file << format_timestamp() << " [" << severity_char(s) << "] [" << loc.file_name()
                  << ":" << loc.line() << "] " << msg << std::endl;
        }
    };
}

// Shared state for all sinks - must be outside template to ensure single instance
static std::mutex& get_sinks_mutex()
{
    static std::mutex m;
    return m;
}

static std::vector<std::optional<sink_entry>>& get_sinks()
{
    static auto sinks = []() {
        // cppcheck-suppress migraphx-RedundantCast
        auto level = static_cast<severity>(
            value_of(MIGRAPHX_LOG_LEVEL{}, static_cast<size_t>(severity::info)));
        return std::vector<std::optional<sink_entry>>{sink_entry{make_stderr_sink(), level}};
    }();
    return sinks;
}

// Thread-safe access to sinks (stderr sink is automatically initialized at index 0)
template <class F>
static void access_sinks(F f)
{
    std::lock_guard<std::mutex> lock(get_sinks_mutex());
    f(get_sinks());
}

size_t add_sink(sink s, severity level)
{
    size_t id = 0;
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        // Find an empty slot or add a new one
        for(size_t i = 0; i < sinks.size(); ++i)
        {
            if(not sinks[i].has_value())
            {
                sinks[i] = sink_entry{std::move(s), level};
                id       = i;
                return;
            }
        }
        id = sinks.size();
        sinks.push_back(sink_entry{std::move(s), level});
    });
    return id;
}

void remove_sink(size_t id)
{
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        if(id < sinks.size())
        {
            sinks[id] = std::nullopt;
        }
    });
}

void set_severity(severity level, size_t id)
{
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        if(id < sinks.size() and sinks[id].has_value())
        {
            sinks[id]->level = level;
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
            if(entry.has_value() and static_cast<size_t>(s) <= static_cast<size_t>(entry->level))
            {
                entry->callback(s, msg, loc);
            }
        }
    });
}

bool is_enabled(severity level)
{
    bool result = false;
    access_sinks([&](std::vector<std::optional<sink_entry>>& sinks) {
        result = std::any_of(sinks.begin(), sinks.end(), [&](const auto& entry) {
            return entry.has_value() and
                   static_cast<size_t>(level) <= static_cast<size_t>(entry->level);
        });
    });
    return result;
}

} // namespace log
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
