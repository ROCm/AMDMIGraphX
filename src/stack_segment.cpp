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
#include <migraphx/stack_segment.hpp>
#include <migraphx/errors.hpp>
#include <cstddef>
#include <utility>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define MIGRAPHX_HAS_ASAN 1
#endif
#elif defined(__SANITIZE_ADDRESS__)
#define MIGRAPHX_HAS_ASAN 1
#endif

#ifdef MIGRAPHX_HAS_ASAN
#include <sanitizer/common_interface_defs.h>
#endif

// Set MIGRAPHX_DISABLE_NATIVE_STACK_SEGMENT to 1 to skip the native stack
// switching implementations and always use the std::thread fallback
#ifndef MIGRAPHX_DISABLE_NATIVE_STACK_SEGMENT
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MIGRAPHX_DISABLE_NATIVE_STACK_SEGMENT 0
#endif

#if MIGRAPHX_DISABLE_NATIVE_STACK_SEGMENT
#define MIGRAPHX_HAS_FIBERS 0
#define MIGRAPHX_HAS_UCONTEXT 0
#elif defined(_WIN32)
#define MIGRAPHX_HAS_FIBERS 1
#define MIGRAPHX_HAS_UCONTEXT 0
#elif defined(__unix__)
#define MIGRAPHX_HAS_FIBERS 0
#define MIGRAPHX_HAS_UCONTEXT 1
#else
#define MIGRAPHX_HAS_FIBERS 0
#define MIGRAPHX_HAS_UCONTEXT 0
#endif

#if MIGRAPHX_HAS_FIBERS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif MIGRAPHX_HAS_UCONTEXT
#include <sys/mman.h>
#include <ucontext.h>
#include <unistd.h>
#else
#include <thread>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {
// Each segment holds max_depth frames of the fix combinator, so it can be much
// smaller than a full thread stack. Pages are only committed when touched.
constexpr std::size_t segment_size = 4 * 1024 * 1024;
} // namespace

#if MIGRAPHX_HAS_FIBERS

namespace {
struct stack_segment
{
    std::function<void()>* f;
    void* caller;
};

void CALLBACK segment_entry(void* p)
{
    auto* s = static_cast<stack_segment*>(p);
    (*s->f)();
    // A fiber entry must never return, so switch back explicitly
    SwitchToFiber(s->caller);
}
} // namespace

void run_on_new_stack(std::function<void()> f)
{
    bool converted = false;
    if(IsThreadAFiber() == FALSE)
    {
        if(ConvertThreadToFiber(nullptr) == nullptr)
            MIGRAPHX_THROW("run_on_new_stack: ConvertThreadToFiber failed");
        converted = true;
    }
    stack_segment s{&f, GetCurrentFiber()};
    void* fiber = CreateFiberEx(0, segment_size, 0, &segment_entry, &s);
    if(fiber == nullptr)
        MIGRAPHX_THROW("run_on_new_stack: CreateFiberEx failed");
    SwitchToFiber(fiber);
    DeleteFiber(fiber);
    if(converted)
        ConvertFiberToThread();
}

#elif MIGRAPHX_HAS_UCONTEXT

namespace {
// makecontext entry functions cant take a pointer argument, so pass the work
// in a thread local instead; the entry runs on the same thread before anything
// else so nested segments cant clobber it early
std::function<void()>& current_segment_func()
{
    thread_local std::function<void()> func;
    return func;
}

struct mapped_stack
{
    void* mem        = MAP_FAILED;
    std::size_t size = 0;

    explicit mapped_stack(std::size_t n)
        : mem(mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)), size(n)
    {
    }
    mapped_stack(const mapped_stack&)            = delete;
    mapped_stack& operator=(const mapped_stack&) = delete;
    ~mapped_stack()
    {
        if(mem != MAP_FAILED)
            munmap(mem, size);
    }
};

void segment_entry()
{
#ifdef MIGRAPHX_HAS_ASAN
    const void* caller_stack      = nullptr;
    std::size_t caller_stack_size = 0;
    __sanitizer_finish_switch_fiber(nullptr, &caller_stack, &caller_stack_size);
#endif
    // Move the work out of the thread local so nested segments can reuse it
    auto f = std::move(current_segment_func());
    f();
#ifdef MIGRAPHX_HAS_ASAN
    // This segment is about to be destroyed, so pass nullptr to free its fake stack
    __sanitizer_start_switch_fiber(nullptr, caller_stack, caller_stack_size);
#endif
    // Returning resumes the caller through uc_link
}
} // namespace

void run_on_new_stack(std::function<void()> f)
{
    auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    // Extra page at the low end becomes a guard page so a segment overflow
    // faults instead of corrupting the heap
    mapped_stack stack{segment_size + page};
    if(stack.mem == MAP_FAILED)
        MIGRAPHX_THROW("run_on_new_stack: mmap failed");
    if(mprotect(stack.mem, page, PROT_NONE) != 0)
        MIGRAPHX_THROW("run_on_new_stack: mprotect failed");

    ucontext_t caller{};
    ucontext_t callee{};
    if(getcontext(&callee) != 0)
        MIGRAPHX_THROW("run_on_new_stack: getcontext failed");
    callee.uc_stack.ss_sp   = static_cast<char*>(stack.mem) + page;
    callee.uc_stack.ss_size = segment_size;
    callee.uc_link          = &caller;
    current_segment_func()  = std::move(f);
    makecontext(&callee, &segment_entry, 0);
#ifdef MIGRAPHX_HAS_ASAN
    void* fake_stack = nullptr;
    __sanitizer_start_switch_fiber(&fake_stack, callee.uc_stack.ss_sp, segment_size);
#endif
    if(swapcontext(&caller, &callee) != 0)
        MIGRAPHX_THROW("run_on_new_stack: swapcontext failed");
#ifdef MIGRAPHX_HAS_ASAN
    __sanitizer_finish_switch_fiber(fake_stack, nullptr, nullptr);
#endif
}

#else

// No stack switching available, so get a fresh stack from a thread instead.
// Unlike the native versions this consumes a kernel task per nested segment.
void run_on_new_stack(std::function<void()> f)
{
    std::thread worker{std::move(f)};
    worker.join();
}

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
