/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#pragma once

#include <hip/hip_runtime_api.h>

#include "logging.hpp"
#include "timer.hpp"

#include <cassert>
#include <half/half.hpp>
#include <iostream>
#include <string>
#include <vector>

#define TIMER_ON 0
#define TRACE_ON 0

#define assertm(exp, msg) assert(((void)msg, exp))

using half = half_float::half;
using namespace half_float::literal;

namespace mlinfer {
struct INoCopy
{
    INoCopy()                          = default;
    virtual ~INoCopy()                 = default;
    INoCopy(const INoCopy&)            = delete;
    INoCopy& operator=(const INoCopy&) = delete;
};

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next  = 0;
    while(next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

#define check_hip_status(hip_call)                                                        \
    do                                                                                    \
    {                                                                                     \
        int status = (hip_call);                                                          \
        if(status != hipSuccess)                                                          \
        {                                                                                 \
            throw std::runtime_error(                                                     \
                "hip error (" + std::to_string(status) +                                  \
                "): " + std::string(hipGetErrorString(static_cast<hipError_t>(status)))); \
        }                                                                                 \
    } while(0);

#define check_hip_status_non_throwing(hip_call)                                                \
    do                                                                                         \
    {                                                                                          \
        int status = (hip_call);                                                               \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            LOG_INFO("hip error (" + std::to_string(status) +                                  \
                     "): " + std::string(hipGetErrorString(static_cast<hipError_t>(status)))); \
        }                                                                                      \
    } while(0);

#define CHECK(condition, error)              \
    do                                       \
    {                                        \
        if(!(condition))                     \
        {                                    \
            std::cerr << error << std::endl; \
        }                                    \
    } while(0);

#if TIMER_ON
#define TIMER_STARTV(s)              \
    static Timer timer##s(#s, true); \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_START(s)         \
    static Timer timer##s(#s); \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_STARTV(s)
#define TIMER_END(s)
#endif

#define TIMED(s, call)  \
    do                  \
    {                   \
        TIMER_START(s); \
        {               \
            call;       \
        }               \
        TIMER_END(s);   \
    } while(0);

} // namespace mlinfer
