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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_LOGGER_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LOGGER_HPP

#include <migraphx/env.hpp>
#include <migraphx/source_location.hpp>
#include <functional>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace log {

enum class severity
{
    none,
    error,
    warn,
    info,
    debug,
    trace
};

using sink = std::function<void(severity, std::string_view, source_location)>;

/**
 * @brief Records a log message. This will invoke the callback for all sinks that are enabled at the given severity.
 *
 * @param s The severity of the log message
 * @param msg The message to log
 * @param loc The source location of the log message
 */
void record(severity s, std::string_view msg, source_location loc = source_location::current());

/**
 * @brief Checks if any sink is enabled at the given severity.
 *
 * @param level The severity to check
 * @return true if any sink is enabled at the given severity, false otherwise
 */
bool is_enabled(severity level);

/**
 * @brief Adds a sink to the logger.
 *
 * @param s The sink to add
 * @param level The severity level of the sink
 * @return The ID of the added sink
 */
size_t add_sink(sink s, severity level = severity::info);

/**
 * @brief Removes a sink from the logger.
 *
 * @param id The ID of the sink to remove
 */
void remove_sink(size_t id);

/**
 * @brief Sets the severity level for a specific sink.
 *
 * @param level The severity level to set
 * @param id The ID of the sink to set the severity for; defaults to 0 for the stderr sink
 */
void set_severity(severity level, size_t id = 0);

/**
 * @brief Adds a file sink to the logger.
 *
 * @param filename The name of the file to log to
 * @param level The severity level of the file logger
 * @return The ID of the added file logger
 */
size_t add_file_logger(std::string_view filename, severity level = severity::info);

template <severity Severity>
struct print
{
    print(source_location ploc = source_location::current()) : loc(ploc) {}

    struct stream
    {
        template <class T>
        stream(severity ps, T&& x, source_location ploc = source_location::current())
            : s(ps), loc(ploc), enabled(is_enabled(s))
        {
            if(enabled)
                ss << x;
        }

        template <class T>
        stream& operator<<(T&& x)
        {
            if(enabled)
                ss << x;
            return *this;
        }

        ~stream()
        {
            if(enabled)
                record(s, ss.str(), loc);
        }

        stream(const stream&)            = delete;
        stream& operator=(const stream&) = delete;

        severity s = severity::none;
        source_location loc;
        bool enabled;
        std::ostringstream ss;
    };

    template <class T>
    stream operator<<(T&& x)
    {
        return stream{Severity, x, loc};
    }

    template <class... Ts>
    void operator()(Ts&&... xs) const
    {
        if(is_enabled(Severity))
        {
            std::ostringstream ss;
            (ss << ... << xs);
            record(Severity, ss.str(), loc);
        }
    }

    print(const print&)            = delete;
    print& operator=(const print&) = delete;

    source_location loc;
};

using error = print<severity::error>;
using warn  = print<severity::warn>;
using info  = print<severity::info>;
using debug = print<severity::debug>;
using trace = print<severity::trace>;

} // namespace log
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
