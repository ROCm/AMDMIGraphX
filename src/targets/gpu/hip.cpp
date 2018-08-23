
#include <migraph/gpu/hip.hpp>

#include <migraph/manage_ptr.hpp>
#include <miopen/miopen.h>

#include <vector>

namespace migraph {
namespace gpu {

using hip_ptr = MIGRAPH_MANAGE_PTR(void, hipFree);

hip_ptr allocate_gpu(std::size_t sz)
{
    void* result;
    // TODO: Check status
    hipMalloc(&result, sz);
    if (result == nullptr)
        throw std::runtime_error("can not allocate GPU memory");
    char * ptr = reinterpret_cast<char*>(result);
    std::cout << "MIGraph allocated mem: [" << result << "," << ptr + sz -1 << "]" << std::endl;
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
    // TODO: Check status
    hipMemcpy(result.data(), x, sz * sizeof(T), hipMemcpyDeviceToHost);
    return result;
}

hip_ptr write_to_gpu(const void* x, std::size_t sz)
{
    auto result = allocate_gpu(sz);
    // TODO: Check status
    hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    return result;
}

migraph::argument allocate_gpu(migraph::shape s)
{
    auto p = share(allocate_gpu(s.bytes()));
    return {s, [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

migraph::argument to_gpu(migraph::argument arg)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes()));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

migraph::argument from_gpu(migraph::argument arg)
{
    migraph::argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

void copy_to_gpu(char* dst, const char* src, std::size_t size)
{
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}
    
} // namespace gpu

} // namespace migraph
