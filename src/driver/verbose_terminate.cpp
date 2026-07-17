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
#include "verbose_terminate.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <typeinfo>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
// Print the active exception's type and what() message, then abort (mirrors libstdc++).
[[noreturn]] void verbose_terminate_handler()
{
    static bool terminating = false;
    if(terminating)
    {
        std::cerr << "terminate called recursively\n";
        std::abort();
    }
    terminating = true;

    if(const std::exception_ptr eptr = std::current_exception())
    {
        try
        {
            std::rethrow_exception(eptr);
        }
        catch(const std::exception& e)
        {
            std::cerr << "terminate called after throwing an instance of '" << typeid(e).name()
                      << "'\n  what():  " << e.what() << '\n';
        }
        catch(...)
        {
            std::cerr << "terminate called after throwing an instance of an unknown exception\n";
        }
    }
    else
    {
        std::cerr << "terminate called without an active exception\n";
    }
    std::abort();
}
} // namespace

void install_verbose_terminate_handler() { std::set_terminate(&verbose_terminate_handler); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
