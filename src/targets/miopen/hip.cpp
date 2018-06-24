
#include <rtg/miopen/hip.hpp>

namespace rtg {
namespace miopen {

hip_ptr gpu_allocate(std::size_t sz)
{
    void* result;
    // TODO: Check status
    hipMalloc(&result, sz);
    return hip_ptr{result};
}

hip_ptr write_to_gpu(const void* x, std::size_t sz)
{
    auto result = gpu_allocate(sz);
    // TODO: Check status
    hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    return result;
}

rtg::argument to_gpu(rtg::argument arg)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes()));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

rtg::argument from_gpu(rtg::argument arg)
{
    rtg::argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

} // namespace miopen

} // namespace rtg
