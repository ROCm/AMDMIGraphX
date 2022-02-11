#ifndef MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP
#define MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

struct assign_none
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        x = y;
    }
};

struct assign_add
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        x += y;
    }
};

struct assign_mul
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR void operator()(T& x, U y) const
    {
        x *= y;
    }
};

template <class T, class U, class V, class W, class F>
__device__ void
scatternd(const T& /* data_t */, const U& indices_t, const V& updates_t, const W& output_t, F f)
{
    auto index         = make_index();
    auto i             = index.global;
    auto updates_shape = updates_t.get_shape();

    if(i < updates_shape.elements())
    {
        auto output_shape = output_t.get_shape();

        auto indices_shape = indices_t.get_shape();
        auto k             = indices_shape.lens.back();
        auto q             = indices_shape.lens.size();

        auto updates_idx = updates_shape.multi(i);
        auto indices_idx = indices_shape.multi(0);
        copy(updates_idx.begin(), updates_idx.begin() + q - 1, indices_idx.begin());

        auto index_start = indices_t.begin() + indices_shape.index(indices_idx);
        auto index_end   = index_start + k;
        auto out_idx     = output_shape.multi(0);
        copy(index_start, index_end, out_idx.begin());
        copy(updates_idx.begin() + q - 1, updates_idx.end(), out_idx.begin() + k);

        f(output_t[out_idx], updates_t[i]);
    }
}

template <class T, class U, class V, class W>
__device__ void scatternd_copy(const T& data_t,
                               const U& /* indices_t */,
                               const V& /* updates_t */,
                               const W& output_t)
{
    auto index        = make_index();
    auto i            = index.global;
    auto output_shape = output_t.get_shape();

    if(i < output_shape.elements())
        output_t[i] = data_t[i];
}

} // namespace migraphx
#endif
