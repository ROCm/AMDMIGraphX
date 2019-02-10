
#include <migraphx/gpu/hip.hpp>

#include <migraphx/manage_ptr.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using hip_ptr = MIGRAPHX_MANAGE_PTR(void, hipFree);

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

template <class T>
std::vector<T> read_from_gpu(const void* x, std::size_t sz)
{
    std::vector<T> result(sz);
    auto status = hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Copy from gpu failed: " + hip_error(status)); // NOLINT
    return result;
}

hip_ptr write_to_gpu(const void* x, std::size_t sz, bool host = false)
{
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

void copy_to_gpu(const argument& src, const argument& dst)
{
    std::size_t src_size = src.get_shape().bytes();
    std::size_t dst_size = dst.get_shape().bytes();
    if(src_size > dst_size)
        MIGRAPHX_THROW("Not enough memory available in destination to do copy");
    auto status = hipMemcpy(dst.data(), src.data(), src_size, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Copy to gpu failed: " + hip_error(status));
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
