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
#include <migraphx/simple_par_for.hpp>

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <process.h>

#include <functional>
#include <memory>
#include <stdexcept>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace detail {

namespace {
unsigned __stdcall run_thread_function(void* arg)
{
    std::unique_ptr<std::function<void()>> f(static_cast<std::function<void()>*>(arg));
    (*f)();
    return 0;
}
} // namespace

void* create_thread_with_stack_size(std::function<void()> f, std::size_t stack_size)
{
    auto fn      = std::make_unique<std::function<void()>>(std::move(f));
    auto* handle = reinterpret_cast<void*>(_beginthreadex(nullptr,
                                                            static_cast<unsigned>(stack_size),
                                                            &run_thread_function,
                                                            fn.get(),
                                                            0,
                                                            nullptr));
    if(handle == nullptr)
        throw std::runtime_error("Failed to create thread"); // NOLINT
    fn.release();
    return handle;
}

void join_thread(void* handle)
{
    WaitForSingleObject(static_cast<HANDLE>(handle), INFINITE);
    CloseHandle(static_cast<HANDLE>(handle));
}

} // namespace detail
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // _WIN32
