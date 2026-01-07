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
#include <migraphx/color.hpp>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::ostream& operator<<(std::ostream& os, const color& c)
{
#ifndef _WIN32
    int fd = -1;
    if(&os == &std::cout)
        fd = STDOUT_FILENO;
    else if(&os == &std::cerr)
        fd = STDERR_FILENO;
    if(fd != -1 && isatty(fd) != 0)
        return os << "\033[" << static_cast<std::size_t>(c) << "m";
#else
    (void)c;
#endif
    return os;
}

std::string colorize(color c, const std::string& s)
{
    std::stringstream ss;
    ss << c << s << color::reset;
    return ss.str();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

