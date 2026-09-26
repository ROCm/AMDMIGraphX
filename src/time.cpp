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
#include <migraphx/time.hpp>
#include <migraphx/errors.hpp>

#ifdef _WIN32
// cppcheck-suppress definePrefix
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <ctime>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

process_cpu_clock::time_point process_cpu_clock::now()
{
#ifdef _WIN32
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    if(GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time) ==
       FALSE)
        MIGRAPHX_THROW("Failed to read the process CPU time");
    auto ticks = [](const FILETIME& t) {
        ULARGE_INTEGER u;
        u.LowPart  = t.dwLowDateTime;
        u.HighPart = t.dwHighDateTime;
        return u.QuadPart;
    };
    // FILETIME counts 100 ns intervals
    using filetime_duration = std::chrono::duration<ULONGLONG, std::ratio<1, 10000000>>;
    return time_point{std::chrono::duration_cast<duration>(
        filetime_duration{ticks(kernel_time) + ticks(user_time)})};
#else
    timespec ts{};
    if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0)
        MIGRAPHX_THROW("Failed to read the process CPU time");
    return time_point{std::chrono::seconds{ts.tv_sec} + std::chrono::nanoseconds{ts.tv_nsec}};
#endif
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
