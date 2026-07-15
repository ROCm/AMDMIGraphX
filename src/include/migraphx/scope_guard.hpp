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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SCOPE_GUARD_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SCOPE_GUARD_HPP

#include <migraphx/config.hpp>
#include <exception>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Runs the action only if the scope is exited via an exception.
template <class F>
struct scope_fail_guard
{
    static_assert(std::is_nothrow_invocable<F&>{}, "scope_fail action must be noexcept");

    F action;
    int uncaught = std::uncaught_exceptions();

    explicit scope_fail_guard(F f) : action(std::move(f)) {}

    scope_fail_guard(const scope_fail_guard&) = delete;

    ~scope_fail_guard()
    {
        if(std::uncaught_exceptions() > uncaught)
            action();
    }
};

template <class F>
scope_fail_guard<F> on_scope_fail(F f)
{
    return scope_fail_guard<F>{std::move(f)};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHX_SCOPE_GUARD_HPP
