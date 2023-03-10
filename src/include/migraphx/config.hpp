/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_CONFIG_HPP
#define MIGRAPHX_GUARD_CONFIG_HPP

#include <migraphx/version.h>
namespace migraphx {

#define MIGRAPHX_CONCAT_1(x, y) x##y
#define MIGRAPHX_CONCAT(x, y) MIGRAPHX_CONCAT_1(x, y)

#define MIGRAPHX_VERSION MIGRAPHX_CONCAT(MIGRAPHX_VERSION_MAJOR, MIGRAPHX_VERSION_MINOR)
#if !defined(MIGRAPHX_USE_CLANG_TIDY) && !defined(DOXYGEN)
#define MIGRAPHX_INLINE_NS MIGRAPHX_CONCAT(VERSION, MIGRAPHX_VERSION)
//#define MIGRAPHX_INLINE_NS VERSION_1
#endif

#ifdef DOXYGEN
#define MIGRAPHX_INLINE_NS internal
#endif

#ifdef MIGRAPHX_USE_CLANG_TIDY
#define MIGRAPHX_TIDY_CONST const
#else
#define MIGRAPHX_TIDY_CONST
#endif

} // namespace migraphx

#endif
