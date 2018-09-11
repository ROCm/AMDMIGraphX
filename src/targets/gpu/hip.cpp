
#include <migraph/gpu/hip.hpp>

#include <migraph/manage_ptr.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraph {
namespace gpu {

using hip_ptr = MIGRAPH_MANAGE_PTR(void, hipFree);

std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

std::size_t get_available_gpu_memory()
{
    size_t free, total;
    auto status = hipMemGetInfo(&free, &total);
    if(status != hipSuccess)
        MIGRAPH_THROW("Failed getting available memory: " + hip_error(status));
    return free;
}

hip_ptr allocate_gpu(std::size_t sz, bool host = false)
{
    if(sz > get_available_gpu_memory())
        MIGRAPH_THROW("Memory not available to allocate buffer: " + std::to_string(sz));
    void* result;
    auto status = host ? hipHostMalloc(&result, sz) : hipMalloc(&result, sz);
    if(status != hipSuccess)
    {
        if(host)
            MIGRAPH_THROW("Gpu allocation failed: " + hip_error(status));
        else
            allocate_gpu(sz, true);
    }
    return hip_ptr{result};
}

template <class T>
hip_ptr write_to_gpu(const T& x)
{
    using type = typename T::value_type;
    auto size  = x.size() * sizeof(type);
    return write_to_gpu(x.data(), size);
}

template <class T>
std::vector<T> read_from_gpu(const void* x, std::size_t sz)
{
    std::vector<T> result(sz);
    auto status = hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    if(status != hipSuccess)
        MIGRAPH_THROW("Copy from gpu failed: " + hip_error(status)); // NOLINT
    return result;
}

hip_ptr write_to_gpu(const void* x, std::size_t sz, bool host = false)
{
    auto result = allocate_gpu(sz, host);
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIGRAPH_THROW("Copy to gpu failed: " + hip_error(status));
    return result;
}

argument allocate_gpu(const shape& s, bool host)
{
    auto p = share(allocate_gpu(s.bytes() + 1, host));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

argument to_gpu(argument arg, bool host)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes(), host));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

argument from_gpu(argument arg)
{
    argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

void gpu_sync() { hipDeviceSynchronize(); }

void copy_to_gpu(char* dst, const char* src, std::size_t size)
{
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}    

} // namespace gpu

} // namespace migraph
