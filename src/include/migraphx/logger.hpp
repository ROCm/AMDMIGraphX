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
#ifndef MIGRAPHX_LOGGER_HPP
#define MIGRAPHX_LOGGER_HPP

#include <migraphx/env.hpp>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_LOG_MIN_LEVEL)

enum class Severity { TRACE, DEBUG, INFO, WARN, ERROR, FATAL };

inline size_t get_min_log_level() {
    static auto log_level = value_of(MIGRAPHX_LOG_MIN_LEVEL{}, 2); // Default to INFO level
    return log_level;
}

class Logger : public std::basic_ostringstream<char> {
public:
    Logger(const char* file, int line, Severity severity) : file_(file), line_(line), severity_(severity) {}
    ~Logger() override 
    {
        std::string message = get_formatted_timestamp(std::chrono::system_clock::now()) + " [" + severity_str[static_cast<size_t>(severity_)] + "] [" + file_ + ":" + std::to_string(line_) + "] " + this->str();
        #ifndef _WIN32
            static const bool use_color = isatty(STDERR_FILENO) != 0;
            if (use_color)
            {
                switch(severity_)
                {
                    case Severity::WARN:
                        message = "\033[33m" + message + "\033[0m"; // Yellow
                        break;
                    case Severity::ERROR: case Severity::FATAL:
                        message = "\033[31m" + message + "\033[0m"; // Red
                        break;
                    case Severity::TRACE: case Severity::DEBUG: case Severity::INFO:
                        break;  // Prevents -Wswitch-enum warning during compilation
                }
            }
        #endif
        std::cerr << message << std::endl;
        if(severity_ == Severity::FATAL)
            std::abort();
    }
private:
    std::string get_formatted_timestamp(std::chrono::time_point<std::chrono::system_clock> time)
    {
        auto now_in_time_t   = std::chrono::system_clock::to_time_t(time);
        auto* now_as_tm_date = std::localtime(&now_in_time_t);
        std::stringstream ss;
        ss << std::put_time(now_as_tm_date, "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    const char* file_;
    int line_;
    Severity severity_;
    const char* severity_str[6] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
};

// NOP stream when log level is disabled
class NopStream {
public:
    template<typename T>
    NopStream& operator<<(const T&) { return *this; }
};

#define LOG_TRACE if(static_cast<size_t>(migraphx::Severity::TRACE) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::TRACE)
#define LOG_DEBUG if(static_cast<size_t>(migraphx::Severity::DEBUG) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::DEBUG)
#define LOG_INFO if(static_cast<size_t>(migraphx::Severity::INFO) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::INFO)
#define LOG_WARN if(static_cast<size_t>(migraphx::Severity::WARN) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::WARN)
#define LOG_ERROR if(static_cast<size_t>(migraphx::Severity::ERROR) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::ERROR)
#define LOG_FATAL if(static_cast<size_t>(migraphx::Severity::FATAL) < migraphx::get_min_log_level()) migraphx::NopStream(); else migraphx::Logger(__FILE__, __LINE__, migraphx::Severity::FATAL)

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
