#ifndef MIGRAPHX_GUARD_KERNELS_PRELOAD_HPP
#define MIGRAPHX_GUARD_KERNELS_PRELOAD_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {

template <class Shape>
constexpr bool is_preloadable()
{
    Shape s{};
    if(not s.broadcasted())
        return false;
}

template <class T, class... Shapes>
constexpr auto traverse_preload(Shapes... ss)
{
    return [=](auto f, auto... g) {
        const index_int max_size = 512 * sizeof(T);
        index_int offset         = 0;
        auto each                = [&](auto x) {
            constexpr auto s    = decltype(x.get_shape()){};
            constexpr auto size = decltype(index_constant<s.element_space()>{}){};
            if(not s.broadcasted())
                return f(x, offset, false_type{});
            if((s.elements() - size) < 64)
                return f(x, offset, false_type{});
            if(offset + size > max_size)
                return f(x, offset, false_type{});
            auto pre_offset = offset;
            offset += size;
            return f(x, pre_offset, true_type{});
        };
        return by(each, g...)(ss...);
    };
}

template <class T, class... Shapes>
constexpr index_int compute_preload_size()
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
                auto b = as_vec(tensor_vec_size(v), buffer);
                for(index_int i = idx.local; i < v.get_shape().element_space(); i += idx.nlocal())
                    b[offset + i] = v.data()[i];
                return x.with(buffer + offset);
            }
            else
            {
                return x;
            }
        },
        invoke);
}

template <class T, class... Ts>
__device__ auto preload(index idx, Ts... xs)
{
    constexpr auto size = compute_preload_size<T, decltype(xs.get_shape())...>();
    return [=](auto f) {
        if constexpr(size > 0)
        {
            __shared__ T buffer[size];
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
