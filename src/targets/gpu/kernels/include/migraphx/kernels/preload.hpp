#ifndef MIGRAPHX_GUARD_KERNELS_PRELOAD_HPP
#define MIGRAPHX_GUARD_KERNELS_PRELOAD_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {

template <class T, class... Shapes>
constexpr auto traverse_preload(Shapes... ss)
{
    return [=](auto f, auto... g) {
        index_int offset = 0;
        auto each        = [&](auto x) {
            constexpr auto s    = decltype(x.get_shape()){};
            constexpr auto size = _c<s.element_space()>;
            if constexpr(not s.broadcasted())
                return f(x, offset, false_type{});
            else if constexpr((s.elements() - size) < 64)
                return f(x, offset, false_type{});
            else
            {
                auto pre_offset = offset;
                offset += size;
                offset += offset % 4;
                return f(x, pre_offset, true_type{});
            }
        };
        return by(each, g...)(ss...);
    };
}

template <class T, class... Shapes>
constexpr index_int compute_preload_size(Shapes...)
{
    index_int size = 0;
    traverse_preload<T>(Shapes{}...)(
        [&](auto s, auto offset, auto) { size = offset + s.element_space(); });
    return size;
}

template <class F, class T, class... Ts>
__device__ auto preload_copy(index idx, F f, __shared__ T* buffer, Ts... xs)
{
    auto invoke = [&](auto... ys) {
        __syncthreads();
        f(ys...);
    };
    traverse_preload<T>(xs...)(
        [&](auto x, auto offset, auto copy) {
            if constexpr(copy)
            {
                auto v = vectorize(x);
                auto b = as_vec(tensor_vec_size(v), buffer + offset);
                idx.local_stride(v.get_shape().element_space(),
                                 [&](auto i) { b[i] = v.data()[i]; });
                return x.with(buffer + offset);
            }
            else
            {
                return x;
            }
        },
        invoke);
}

template <class T>
struct remove_vec
{
    using type = T;
};

template <class T, index_int N>
struct remove_vec<vec<T, N>>
{
    using type = T;
};

template <class T, class... Ts>
__device__ auto preload(index idx, Ts... xs)
{
    using type               = typename remove_vec<T>::type;
    constexpr auto size      = compute_preload_size<type>(xs.get_shape()...);
    const index_int max_size = 512 * sizeof(type);
    return [=](auto f) {
        if constexpr(size > 0 and size < max_size)
        {
            __shared__ type buffer[size];
            preload_copy(idx, f, buffer, xs...);
        }
        else
        {
            f(xs...);
        }
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PRELOAD_HPP
