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

#include <iostream>

namespace mlinfer {

#define LOGGING_OFF 0
#define ENABLE_TIMED_LOGGING 0
#define ENABLE_DEBUG_LOGGING 0

#if(!LOGGING_OFF)
#define LOG_INFO(...)                          \
    do                                         \
    {                                          \
        std::cout << __VA_ARGS__ << std::endl; \
    } while(0)
#define LOG_ERROR(...)                         \
    do                                         \
    {                                          \
        std::cerr << __VA_ARGS__ << std::endl; \
    } while(0)
#define LOG_STATE(...)                                                                \
    do                                                                                \
    {                                                                                 \
        std::cout << "================================================" << std::endl; \
        std::cout << __VA_ARGS__ << std::endl;                                        \
        std::cout << "================================================" << std::endl; \
    } while(0)
#else
#define LOG_INFO(...) (void)0
#define LOG_ERROR(...) (void)0
#define LOG_STATE(...) (void)0
#endif

#if(ENABLE_TIMED_LOGGING || ENABLE_DEBUG_LOGGING)
#define LOG_TIMED(...) LOG_INFO(__VA_ARGS__)
#else
#define LOG_TIMED(...) (void)0
#endif

#if ENABLE_DEBUG_LOGGING
#define LOG_DEBUG(...) LOG_INFO(__VA_ARGS__)
#else
#define LOG_DEBUG(...) (void)0
#endif

} // namespace mlinfer
