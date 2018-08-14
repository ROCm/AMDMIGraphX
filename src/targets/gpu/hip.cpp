
#include <migraph/gpu/hip.hpp>

#include <migraph/manage_ptr.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraph {
namespace gpu {

using hip_ptr = MIGRAPH_MANAGE_PTR(void, hipFree);

std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

hip_ptr allocate_gpu(std::size_t sz)
{
    void* result;
    auto status = hipMalloc(&result, sz);
    if(status != hipSuccess)
        MIGRAPH_THROW("Gpu allocation failed: " + hip_error(status));
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

hip_ptr write_to_gpu(const void* x, std::size_t sz)
{
    auto result = allocate_gpu(sz);
    auto status = hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    if(status != hipSuccess)
        MIGRAPH_THROW("Copy to gpu failed: " + hip_error(status));
    return result;
}

argument allocate_gpu(shape s)
{
    auto p = share(allocate_gpu(s.bytes() + 1));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

argument to_gpu(argument arg)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes()));
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

} // namespace gpu

} // namespace migraph
