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

    template<class T, class InnerLens, class OuterLens>
    static constexpr auto slice(T x, index_int group, InnerLens, OuterLens)
    {
        constexpr auto outer_strides = transform_i(x.get_shape().strides, [&](auto stride, auto i) {
            constexpr auto inner_lens = InnerLens{};
            constexpr auto outer_lens = OuterLens{};
            if (inner_lens[i] == outer_lens[i])
                return stride;
            return stride * inner_lens[i];
        });
        constexpr auto is = make_shape(InnerLens{}, x.get_shape().strides);
        constexpr auto os = make_shape(OuterLens{}, outer_strides);
        auto offset = os.index(group);
        return make_tensor_view(x.data() + offset, is);
    }

    template <class InnerShape, class OuterShape>
    static __device__ auto auto_slice(index idx)
    {
        return make_transform([=](auto f, auto... xs) {
            idx.group_stride(OuterShape{}.elements(),
                             [=](auto group) {
                                f(slice(xs, group, InnerShape{}.lens, OuterShape{}.lens)...); 
                            });
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

template <class... Mode, class InnerShape, class OuterShape>
__device__ auto auto_tile(InnerShape, OuterShape)
{
    if constexpr((is_same<Mode, tile::none>{} and ...))
    {
        return transform_args();
    }
    else
    {
        auto idx = make_index();
        return transform_args(tile::auto_slice<InnerShape, OuterShape>(idx),
                              auto_prestore<is_same<Mode, tile::store>{}...>(idx),
                              auto_preload<is_same<Mode, tile::load>{}...>(idx));
    }
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_TILE_HPP
