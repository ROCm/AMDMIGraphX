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
#include <migraphx/errors.hpp>
#include <migraphx/gpu/pack_args.hpp>
#include <algorithm>
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
// HIP_LAUNCH_PARAM_* expand to C-style pointer casts (((void*)0x01) etc.) that
// clang-tidy rejects, so the launch-config sentinels are produced here behind
// that workaround; everything else refers to these accessors. Functions (rather
// than globals) also keep clang-tidy from flagging a non-const global pointer.
#ifdef MIGRAPHX_USE_CLANG_TIDY
void* launch_param_pointer() { return nullptr; }
void* launch_param_size() { return nullptr; }
void* launch_param_end() { return nullptr; }
#else
void* launch_param_pointer() { return HIP_LAUNCH_PARAM_BUFFER_POINTER; }
void* launch_param_size() { return HIP_LAUNCH_PARAM_BUFFER_SIZE; }
void* launch_param_end() { return HIP_LAUNCH_PARAM_END; }
#endif
} // namespace

std::array<void*, 5> pack_kernel_config(char* buffer, std::size_t* size)
{
    return {launch_param_pointer(), buffer, launch_param_size(), size, launch_param_end()};
}

std::vector<char> unpack_kernel_config(void** extra)
{
    if(extra == nullptr)
        return {};
    char* buffer     = nullptr;
    std::size_t size = 0;
    bool has_size    = false;
    // The config is a sequence of (tag, value) pairs terminated by the end
    // sentinel; the fixed cap guards against an unterminated array.
    constexpr std::size_t max_tokens = 16;
    for(std::size_t i = 0; i < max_tokens and extra[i] != launch_param_end(); i += 2)
    {
        if(extra[i] == launch_param_pointer())
            buffer = static_cast<char*>(extra[i + 1]);
        else if(extra[i] == launch_param_size())
        {
            size     = *static_cast<std::size_t*>(extra[i + 1]);
            has_size = true;
        }
        else
            return {};
    }
    if(buffer == nullptr or not has_size)
        return {};
    return {buffer, buffer + size};
}

std::vector<std::pair<std::size_t, char*>>
unpack_kernel_config(const std::vector<char>& buffer,
                     const std::map<std::size_t, kernel_argument_value>& kernel_args)
{
    std::vector<std::pair<std::size_t, char*>> pointers;
    auto record = [&](std::size_t off) {
        if(off + sizeof(char*) > buffer.size())
            return;
        pointers.emplace_back(off, read_pointer(buffer.data() + off));
    };
    if(kernel_args.empty())
    {
        // The all-pointer launch path packs one device pointer per 8-byte word.
        for(std::size_t off = 0; off + sizeof(char*) <= buffer.size(); off += sizeof(char*))
            record(off);
        return pointers;
    }
    // Replay the pack_args layout (code_object_op::finalize): a pointer slot has
    // empty data (8 bytes, align 8); a scalar carries its own bytes and alignment.
    std::size_t pos = 0;
    for(const auto& [idx, v] : kernel_args)
    {
        bool is_pointer   = v.data.empty();
        std::size_t align = is_pointer ? sizeof(char*) : v.align;
        std::size_t size  = is_pointer ? sizeof(char*) : v.data.size();
        assert(align != 0); // a scalar argument always carries a real alignment
        pos += pack_padding(pos, align);
        if(is_pointer)
            record(pos);
        pos += size;
    }
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
