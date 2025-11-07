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
#include <sstream>

namespace migraphx {
namespace log {
inline namespace MIGRAPHX_INLINE_NS {

enum class severity
{
    NONE,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE
};

void record(severity s, std::string_view msg, source_location loc = source_location::current());

bool is_enabled(severity s);

void set_log_level(severity s);

void add_file_logger(std::string_view filename, severity s = severity::INFO);

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

        severity s = severity::NONE;
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

using error = print<severity::ERROR>;
using warn  = print<severity::WARN>;
using info  = print<severity::INFO>;
using debug = print<severity::DEBUG>;
using trace = print<severity::TRACE>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace log
} // namespace migraphx
#endif
