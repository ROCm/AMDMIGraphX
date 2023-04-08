#ifndef MIGRAPHX_GUARD_KERNELS_GEMM_BATCHER_HPP
#define MIGRAPHX_GUARD_KERNELS_GEMM_BATCHER_HPP

#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/index.hpp>

namespace migraphx {

template <class Tensor>
constexpr auto gemm_get_batches()
{
    constexpr auto lens     = get_shape_c<Tensor>{}.lens;
    constexpr auto strides  = get_shape_c<Tensor>{}.strides;
    constexpr auto new_lens = sequence(
        lens.size() - _c<2>, [&](auto... is) { return make_const_array(_c<lens[is]>...); });
    constexpr auto new_strides = sequence(
        strides.size() - _c<2>, [&](auto... is) { return make_const_array(_c<strides[is]>...); });
    return make_shape(new_lens, new_strides);
}

template <class Tensor>
constexpr auto gemm_get_matrix()
{
    constexpr auto lens        = get_shape_c<Tensor>{}.lens;
    constexpr auto strides     = get_shape_c<Tensor>{}.strides;
    constexpr auto m           = lens.size() - _c<2>;
    constexpr auto n           = lens.size() - _c<1>;
    constexpr auto new_lens    = make_const_array(_c<lens[m]>, _c<lens[n]>);
    constexpr auto new_strides = make_const_array(_c<strides[m]>, _c<strides[n]>);
    return make_shape(new_lens, new_strides);
}

template <class Tensor, class T>
constexpr auto gemm_batch_slice(Tensor t, T i)
{
    constexpr auto batch  = gemm_get_batches<Tensor>();
    constexpr auto matrix = gemm_get_matrix<Tensor>();
    return make_tensor_view(t.data() + batch.index(i), matrix);
}

template <class BlocksPerBatch, class T, class... Ts>
constexpr auto gemm_batch_args(index idx, BlocksPerBatch bpb, T x, Ts... xs)
{
    return [=](auto f) {
        // All tensors should have the same rank
        static_assert(
            (true and ... and (get_shape_c<T>{}.lens.size() == get_shape_c<Ts>{}.lens.size())));
        if constexpr(get_shape_c<T>{}.lens.size() > 2)
        {
            // Get the first batch since all batches should have the same number of elements
            constexpr auto batch = gemm_get_batches<T>();
            static_assert(
                (true and ... and (batch.elements() == gemm_get_batches<Ts>().elements())));
            idx.group_stride(bpb * batch.elements(), [&](auto gidx) {
                const auto batch_idx = gidx / bpb;
                f(gemm_batch_slice(x, batch_idx), gemm_batch_slice(xs, batch_idx)...);
            });
        }
        else
        {
            f(x, xs...);
        }
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_GEMM_BATCHER_HPP
