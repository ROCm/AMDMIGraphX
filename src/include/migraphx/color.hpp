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
#ifndef MIGRAPHX_GUARD_RTGLIB_COLOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_COLOR_HPP

#include <iostream>
#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

enum class color
{
    reset      = 0,
    bold       = 1,
    underlined = 4,
    fg_red     = 31,
    fg_green   = 32,
    fg_yellow  = 33,
    fg_blue    = 34,
    fg_cyan    = 36,
    fg_white   = 37,
    fg_default = 39,
    bg_red     = 41,
    bg_green   = 42,
    bg_yellow  = 43,
    bg_blue    = 44,
    bg_default = 49
};

MIGRAPHX_EXPORT std::ostream& operator<<(std::ostream& os, const color& c);

MIGRAPHX_EXPORT std::string colorize(color c, const std::string& s);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
