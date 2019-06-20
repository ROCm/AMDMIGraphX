
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_VISIT_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_VISIT_HPP

#include <migraphx/gpu/device/tensor_view.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class F>
void visit_tensor_size(std::size_t n, F f)
{
    switch(n)
    {
    case 1:
    {
        f(std::integral_constant<std::size_t, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<std::size_t, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<std::size_t, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<std::size_t, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<std::size_t, 5>{});
        break;
    }
    default: throw std::runtime_error("Unknown tensor size");
    }
}

inline std::size_t tensor_size(const shape& x) { return x.lens().size(); }

template <class T>
auto tensor_size(const T& x) -> decltype(x.get_shape().lens().size())
{
    return x.get_shape().lens().size();
}

template <class T, class... Ts>
auto hip_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        visit_tensor_size(tensor_size(x), [&](auto dim) {
            visit_all(x, xs...)([&](auto... vs) { f(make_hip<dim>(device_cast(vs))...); });
        });
    };
}

template <std::size_t N, class T, class... Ts>
auto hip_vec_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        visit_tensor_size(tensor_size(x), [&](auto dim) {
            visit_all(x,
                      xs...)([&](auto... vs) { f(make_hip<dim>(as_vec<N>(device_cast(vs)))...); });
        });
    };
}

template <class T, class... Ts>
auto hip_pointer_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) { visit_all(x, xs...)([&](auto... vs) { f(device_cast(vs.data())...); }); };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
