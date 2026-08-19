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
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <algorithm>
#include <array>
#include <cassert>

#ifdef _WIN32
#include <hip/hip_ext.h>
#else
// extern declare the function since hip/hip_ext.h header is broken
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t, // NOLINT
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           uint32_t,
                                           size_t,
                                           hipStream_t,
                                           void**,
                                           void**,
                                           hipEvent_t = nullptr,
                                           hipEvent_t = nullptr,
                                           uint32_t   = 0);
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

extern std::string hip_error(int error);

using hip_module_ptr = MIGRAPHX_MANAGE_PTR(hipModule_t, hipModuleUnload);

struct kernel_impl
{
    hip_module_ptr module = nullptr;
    hipFunction_t fun     = nullptr;
};

static hip_module_ptr load_module(const char* image)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, image);
    hip_module_ptr m{raw_m};
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to load module: " + hip_error(status));
    return m;
}

kernel::kernel(const char* image, const std::string& name) : impl(std::make_shared<kernel_impl>())
{
    impl->module = load_module(image);
    auto status  = hipModuleGetFunction(&impl->fun, impl->module.get(), name.c_str());
    if(hipSuccess != status)
        MIGRAPHX_THROW("Failed to get function: " + name + ": " + hip_error(status));
}

bool kernel::empty() const { return impl == nullptr; }

hipFunction_t kernel::get_function() const { return impl == nullptr ? nullptr : impl->fun; }

namespace {
// HIP_LAUNCH_PARAM_* expand to C-style pointer casts that clang-tidy rejects,
// so the launch-config sentinels are produced behind these accessors
// (functions rather than globals to avoid the non-const-global lint).
#ifdef MIGRAPHX_USE_CLANG_TIDY
void* launch_param_pointer() { return nullptr; }
void* launch_param_size() { return nullptr; }
void* launch_param_end() { return nullptr; }
#else
void* launch_param_pointer() { return HIP_LAUNCH_PARAM_BUFFER_POINTER; }
void* launch_param_size() { return HIP_LAUNCH_PARAM_BUFFER_SIZE; }
void* launch_param_end() { return HIP_LAUNCH_PARAM_END; }
#endif

// A launch-config entry: (tag, value).
using launch_param = std::pair<void*, void*>;
static_assert(sizeof(launch_param) == 2 * sizeof(void*),
              "the extra config array is viewed as an array of launch_param pairs");

// The value of the entry with `tag` in [first, last), or nullptr when absent.
void* find_launch_param(const launch_param* first, const launch_param* last, void* tag)
{
    auto it = std::find_if(first, last, [&](const launch_param& p) { return p.first == tag; });
    return it == last ? nullptr : it->second;
}
} // namespace

std::array<void*, 5> pack_kernel_config(char* buffer, std::size_t* size)
{
    return {launch_param_pointer(), buffer, launch_param_size(), size, launch_param_end()};
}

std::vector<char> unpack_kernel_config(void** extra)
{
    if(extra == nullptr)
        return {};
    // The config is a sequence of (tag, value) pairs terminated by the end
    // sentinel; an array with no sentinel within the fixed cap is rejected.
    constexpr std::size_t max_params = 8;
    const auto* first                = reinterpret_cast<const launch_param*>(extra);
    const auto* last = std::find_if(first, first + max_params, [](const launch_param& p) {
        return p.first == launch_param_end();
    });
    if(last == first + max_params)
        return {};
    if(std::any_of(first, last, [](const launch_param& p) {
           return p.first != launch_param_pointer() and p.first != launch_param_size();
       }))
        return {};
    auto* buffer = static_cast<char*>(find_launch_param(first, last, launch_param_pointer()));
    auto* size   = static_cast<std::size_t*>(find_launch_param(first, last, launch_param_size()));
    if(buffer == nullptr or size == nullptr)
        return {};
    return {buffer, buffer + *size};
}

void write_pointer(char* pos, const char* p)
{
    auto bytes = migraphx::bit_cast<std::array<char, sizeof(char*)>>(p);
    std::copy(bytes.begin(), bytes.end(), pos);
}

static char* read_pointer(const char* pos)
{
    std::array<char, sizeof(char*)> bytes{};
    std::copy(pos, pos + sizeof(char*), bytes.begin());
    return migraphx::bit_cast<char*>(bytes);
}

std::vector<std::pair<std::size_t, char*>>
unpack_pointer_args(const std::vector<char>& buffer,
                    const std::map<std::size_t, kernel_argument_value>& kernel_args)
{
    std::vector<std::pair<std::size_t, char*>> pointers;
    if(kernel_args.empty())
    {
        // The all-pointer launch path packs one device pointer per 8-byte word.
        for(std::size_t off = 0; off + sizeof(char*) <= buffer.size(); off += sizeof(char*))
            pointers.emplace_back(off, read_pointer(buffer.data() + off));
        return pointers;
    }
    // A slot past the end of the buffer is skipped rather than read.
    for_each_kernarg_slot(kernel_args, [&](std::size_t pos, bool is_pointer) {
        if(is_pointer and pos + sizeof(char*) <= buffer.size())
            pointers.emplace_back(pos, read_pointer(buffer.data() + pos));
    });
    return pointers;
}

static void launch_kernel(hipFunction_t fun,
                          hipStream_t stream,
                          std::size_t global,
                          std::size_t local,
                          void* kernargs,
                          std::size_t size,
                          hipEvent_t start,
                          hipEvent_t stop)
{
    assert(global > 0);
    assert(local > 0);
    auto config = pack_kernel_config(static_cast<char*>(kernargs), &size);

    auto status = hipExtModuleLaunchKernel(
        fun, global, 1, 1, local, 1, 1, 0, stream, nullptr, config.data(), start, stop);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to launch kernel: " + hip_error(status));
    if(stop != nullptr)
    {
        status = hipEventSynchronize(stop);
        if(status != hipSuccess)
            MIGRAPHX_THROW("Failed to sync event: " + hip_error(status));
    }
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    pointers args,
                    hipEvent_t start,
                    hipEvent_t stop) const
{
    assert(impl != nullptr);
    void* kernargs   = reinterpret_cast<void*>(args.data());
    std::size_t size = args.bytes();

    launch_kernel(impl->fun, stream, global, local, kernargs, size, start, stop);
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    const std::vector<kernel_argument>& args,
                    hipEvent_t start,
                    hipEvent_t stop) const
{
    assert(impl != nullptr);
    std::vector<char> kernargs = pack_args(args);
    std::size_t size           = kernargs.size();

    launch_kernel(impl->fun, stream, global, local, kernargs.data(), size, start, stop);
}

void kernel::launch(hipStream_t stream,
                    std::size_t global,
                    std::size_t local,
                    void* kernargs,
                    std::size_t kernargs_size,
                    hipEvent_t start,
                    hipEvent_t stop) const
{
    assert(impl != nullptr);
    launch_kernel(impl->fun, stream, global, local, kernargs, kernargs_size, start, stop);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
