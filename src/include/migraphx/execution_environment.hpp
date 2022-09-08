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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_EXECUTION_ENV_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_EXECUTION_ENV_HPP

#include <migraphx/any_ptr.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct execution_environment
{
    execution_environment() = default;
    execution_environment(any_ptr q, bool is_async): queue(q), async(is_async){};
    execution_environment(void* q, bool is_async)
        : queue(any_ptr(q, "hipStream_t")), async(is_async){};

    any_ptr queue;
    bool async = false;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif /* MIGRAPHX_GUARD_MIGRAPHLIB_EXECUTION_ENV_HPP */
