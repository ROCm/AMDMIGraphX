
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

inline shape get_shape(const shape& x) { return x; }

template <class T>
auto get_shape(const T& x) -> decltype(x.get_shape())
{
    return x.get_shape();
}

template <class V, class F, class... Ts>
void hip_visit_all_impl(const shape& s, F f, V&& v, Ts&&... xs)
{
    visit_tensor_size(s.lens().size(),
                      [&](auto ndim) { s.visit_type([&](auto as) { v(f(xs, ndim, as)...); }); });
}

template <class F>
struct hip_convert
{
    F f;
    template <class RawData, class N, class As>
    auto operator()(RawData x, N ndim, As as) const
        -> decltype(make_hip_view<ndim>(x.get_shape(), f(as.from(x.data()))))
    {
        return make_hip_view<ndim>(x.get_shape(), f(as.from(x.data())));
    }

    template <class N, class As>
    auto operator()(const shape& s, N ndim, As) const
    {
        return make_hip_shape<ndim>(s);
    }
};

template <class F>
hip_convert<F> make_hip_convert(F f)
{
    return {f};
}

template <class T, class... Ts>
auto hip_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        hip_visit_all_impl(
            get_shape(x), make_hip_convert([](auto* p) { return device_cast(p); }), f, x, xs...);
    };
}

template <std::size_t N, class T, class... Ts>
auto hip_vec_visit_all(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        hip_visit_all_impl(get_shape(x),
                           make_hip_convert([](auto* p) { return as_vec<N>(device_cast(p)); }),
                           f,
                           x,
                           xs...);
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
