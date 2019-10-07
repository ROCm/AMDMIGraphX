
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_VISIT_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_VISIT_HPP

#include <migraphx/gpu/device/tensor_view.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class F>
void visit_tensor_size(index_int n, F f)
{
    switch(n)
    {
    case 1:
    {
        f(std::integral_constant<index_int, 1>{});
        break;
    }
    case 2:
    {
        f(std::integral_constant<index_int, 2>{});
        break;
    }
    case 3:
    {
        f(std::integral_constant<index_int, 3>{});
        break;
    }
    case 4:
    {
        f(std::integral_constant<index_int, 4>{});
        break;
    }
    case 5:
    {
        f(std::integral_constant<index_int, 5>{});
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
    std::initializer_list<migraphx::shape::type_t> types = {get_shape(xs).type()...};
    if(!std::all_of(
           types.begin(), types.end(), [&](migraphx::shape::type_t t) { return t == s.type(); }))
        MIGRAPHX_THROW("Types must be the same");
    std::initializer_list<index_int> ranks = {
        static_cast<index_int>(get_shape(xs).lens().size())...};
    if(!std::all_of(ranks.begin(), ranks.end(), [&](index_int r) { return r == s.lens().size(); }))
        MIGRAPHX_THROW("Ranks must be the same");
    visit_tensor_size(s.lens().size(),
                      [&](auto ndim) { s.visit_type([&](auto as) { v(f(xs, ndim, as)...); }); });
}

template <class V, class F, class... Ts>
void hip_visit_views_impl(const shape& s, F f, V&& v, Ts&&... xs)
{
    std::initializer_list<index_int> ranks = {
        static_cast<index_int>(get_shape(xs).lens().size())...};
    if(!std::all_of(ranks.begin(), ranks.end(), [&](index_int r) { return r == s.lens().size(); }))
        MIGRAPHX_THROW("Ranks must be the same");
    visit_tensor_size(s.lens().size(), [&](auto ndim) { v(f(xs, ndim)...); });
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

template <class F>
struct hip_convert_view
{
    F f;
    template <class T, class N>
    auto operator()(tensor_view<T> x, N ndim) const
    {
        return make_hip_view<ndim>(f(x));
    }

    template <class N>
    auto operator()(const shape& s, N ndim) const
    {
        return make_hip_shape<ndim>(s);
    }
};

template <class F>
hip_convert_view<F> make_hip_convert_view(F f)
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

template <index_int N, class T, class... Ts>
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

template <class T, class... Ts>
auto hip_visit_views(T&& x, Ts&&... xs)
{
    return [&](auto f) {
        hip_visit_views_impl(get_shape(x),
                             make_hip_convert_view([](auto v) { return device_cast(v); }),
                             f,
                             x,
                             xs...);
    };
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
