#ifndef MIGRAPHX_GUARD_KERNELS_TILE_HPP
#define MIGRAPHX_GUARD_KERNELS_TILE_HPP

#include <migraphx/kernels/prestore.hpp>
#include <migraphx/kernels/preload.hpp>

namespace migraphx {

struct tile
{
    struct load
    {
    };
    struct store
    {
    };
    struct none{};

    static constexpr auto outer()
    {
        return [](auto axis, auto a) {
            return transform_i(a, [=](auto x, auto i) {
                if constexpr(i <= axis)
                    return x;
                else
                    return 1;
            });
        };
    }

    static constexpr auto inner()
    {
        return [](auto axis, auto a) {
            return transform_i(a, [=](auto x, auto i) {
                if constexpr(i > axis)
                    return x;
                else
                    return 1;
            });
        };
    }

    template <index_int Axis, class Select, class Shape>
    static constexpr auto slice(Select select, Shape)
    {
        constexpr Shape s{};
        return make_shape(select(_c<Axis>, s.lens), select(_c<Axis>, s.strides));
    }

    template <index_int Axis, class T>
    static constexpr auto slice_tensor(index_int i, T x)
    {
        constexpr auto s = get_shape_c<T>{};
        auto offset      = slice<Axis>(outer(), s).index(i);
        return make_tensor_view(x.data() + offset, slice<Axis>(inner(), s));
    }

    template <index_int Axis, class T, class... Ts>
    static constexpr auto get_size(T, Ts...)
    {
        // TODO: Assert all slices are the same size
        constexpr auto size = slice<Axis>(outer(), get_shape_c<T>{}).elements();
        return size;
    }

    template <index_int Axis>
    static __device__ auto auto_slice(index idx)
    {
        return make_transform([=](auto f, auto... xs) {
            idx.group_stride(get_size<Axis>(xs...),
                             [=](auto group) { f(slice_tensor<Axis>(group, xs)...); });
        });
    }
};

template<bool Tiled>
__device__ auto tile_stride(index idx)
{
    if constexpr(Tiled)
    {
        return [=](auto... xs) { return idx.local_stride(xs...); };
    }
    else
    {
        return [=](auto... xs) { return idx.global_stride(xs...); };
    }
}

template <index_int Axis, class... Mode>
__device__ auto auto_tile()
{
    if constexpr((is_same<Mode, tile::none>{} and ...))
    {
        return transform_args();
    }
    else
    {
        auto idx = make_index();
        return transform_args(tile::auto_slice<Axis>(idx),
                              auto_prestore<is_same<Mode, tile::store>{}...>(idx),
                              auto_preload<is_same<Mode, tile::load>{}...>(idx));
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TILE_HPP
