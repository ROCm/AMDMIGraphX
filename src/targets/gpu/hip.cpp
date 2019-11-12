
#include <migraphx/gpu/hip.hpp>

#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/contiguous.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using hip_ptr = MIGRAPHX_MANAGE_PTR(void, hipFree);
using hip_host_ptr = MIGRAPHX_MANAGE_PTR(void, hipHostUnregister);

std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

std::size_t get_available_gpu_memory()
{
    size_t free;
    size_t total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed getting available memory: " + hip_error(status));
    return free;
}

void* get_device_ptr(void* hptr)
{
    void* result = nullptr;
    auto status = hipHostGetDevicePointer(&result, hptr, 0);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed getting device pointer: " + hip_error(status));
    return result;
}

hip_ptr allocate_gpu(std::size_t sz, bool host = false)
{
    if(sz > get_available_gpu_memory())
        MIGRAPHX_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = host ? hipHostMalloc(&result, sz) : hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        if(host)
            MIGRAPHX_THROW("Gpu allocation failed: " + hip_error(status));
        else
            allocate_gpu(sz, true);
    }
    return hip_ptr{result};
}

hip_host_ptr register_on_gpu(void* ptr, std::size_t sz)
{
    auto status = hipHostRegister(ptr, sz, hipHostRegisterMapped);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Gpu register failed: " + hip_error(status));

    return hip_host_ptr{ptr};
}

template <class T>
std::vector<T> read_from_gpu(const void* x, std::size_t sz)
{
    gpu_sync();
    std::vector<T> result(sz);
    auto status = hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Copy from gpu failed: " + hip_error(status)); // NOLINT
    return result;
}

hip_ptr write_to_gpu(const void* x, std::size_t sz, bool host = false)
{
    gpu_sync();
    auto result = allocate_gpu(sz, host);
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Copy to gpu failed: " + hip_error(status));
    return result;
}

template <class T>
hip_ptr write_to_gpu(const T& x)
{
    using type = typename T::value_type;
    auto size  = x.size() * sizeof(type);
    return write_to_gpu(x.data(), size);
}

argument allocate_gpu(const shape& s, bool host)
{
    auto p = share(allocate_gpu(s.bytes() + 1, host));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

argument register_on_gpu(argument arg)
{
    auto p = share(register_on_gpu(arg.data(), arg.get_shape().bytes()));
    return {arg.get_shape(), [p, a = std::move(arg)]() mutable { return reinterpret_cast<char*>(get_device_ptr(p.get())); }};
}

argument to_gpu(const argument& arg, bool host)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes(), host));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

argument from_gpu(const argument& arg)
{
    argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

void set_device(std::size_t id)
{
    auto status = hipSetDevice(id);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Error setting device");
}

void gpu_sync() { hipDeviceSynchronize(); }

void hip_async_copy(context& ctx, const argument& src, const argument& dst, hipMemcpyKind kind)
{
    std::size_t src_size = src.get_shape().bytes();
    std::size_t dst_size = dst.get_shape().bytes();
    if(src_size > dst_size)
        MIGRAPHX_THROW("Not enough memory available in destination to do copy");
    auto status = hipMemcpyAsync(dst.data(), src.data(), src_size, kind, ctx.get_stream().get());
    if(status != hipSuccess)
        MIGRAPHX_THROW("Gpu copy failed: " + hip_error(status));
}

void gpu_copy(context& ctx, const argument& src, const argument& dst)
{
    // Workaround: Use contiguous as hip's memcpy is broken
    device::contiguous(ctx.get_stream().get(), dst, src);
    // hip_async_copy(ctx, src, dst, hipMemcpyDeviceToDevice);
}

void copy_to_gpu(context& ctx, const argument& src, const argument& dst)
{
    gpu_copy(ctx, register_on_gpu(src), dst);
}

void copy_from_gpu(context& ctx, const argument& src, const argument& dst)
{
    gpu_copy(ctx, src, register_on_gpu(dst));
}

argument get_preallocation(context& ctx, std::string id)
{
    return ctx.get_current_device().preallocations.at(id);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
