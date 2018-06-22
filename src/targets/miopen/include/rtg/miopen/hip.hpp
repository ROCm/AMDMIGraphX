#ifndef RTG_GUARD_RTGLIB_HIP_HPP
#define RTG_GUARD_RTGLIB_HIP_HPP

#include <rtg/manage_ptr.hpp>

#include <miopen/miopen.h>

namespace rtg {
namespace miopen {

using hip_ptr = RTG_MANAGE_PTR(void, hipFree);

inline hip_ptr gpu_allocate(std::size_t sz)
{
    void* result;
    // TODO: Check status
    hipMalloc(&result, sz);
    return hip_ptr{result};
}

inline hip_ptr write_to_gpu(const void* x, std::size_t sz)
{
    auto result = gpu_allocate(sz);
    // TODO: Check status
    hipMemcpy(result.get(), x, sz, hipMemcpyHostToDevice);
    return result;
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

inline rtg::argument to_gpu(rtg::argument arg)
{
    auto p = share(write_to_gpu(arg.data(), arg.get_shape().bytes()));
    return {arg.get_shape(), [p]() mutable { return reinterpret_cast<char*>(p.get()); }};
}

inline rtg::argument from_gpu(rtg::argument arg)
{
    rtg::argument result;
    arg.visit([&](auto x) {
        using type = typename decltype(x)::value_type;
        auto v     = read_from_gpu<type>(arg.data(), x.get_shape().bytes() / sizeof(type));
        result     = {x.get_shape(), [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });
    return result;
}

struct hip_allocate
{
    std::string name() const { return "hip::allocate"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return inputs.front();
    }
    argument compute(shape output_shape, std::vector<argument>) const
    {
        char* data = nullptr;
        // TODO: Check return status
        hipMalloc(&data, output_shape.bytes());
        return {output_shape, data};
    }
};

struct hip_free
{
    std::string name() const { return "hip::free"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        return {};
    }
    argument compute(shape, std::vector<argument> args) const
    {
        // TODO: Check return status
        hipFree(args.front().data());
        return {};
    }
};

} // namespace miopen

} // namespace rtg

#endif
