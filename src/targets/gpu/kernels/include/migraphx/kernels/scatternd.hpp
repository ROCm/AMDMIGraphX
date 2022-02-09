#ifndef MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP
#define MIGRAPHX_GUARD_KERNELS_SCATTERND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/dfor.hpp>
#include <migraphx/kernels/basic_ops.hpp>
#include <args.hpp>

namespace migraphx {

template <class T1, class T2>
struct scatternd_settings
{
    T1 is_add{};
    T2 is_mul{};
};

template <class... Ts>
constexpr scatternd_settings<Ts...> make_scatternd_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class V, class W, class Settings>
__device__ void
scatternd(const T& data_t, const U& indices_t, const V& updates_t, const W& output_t, Settings s)
{
    auto index           = make_index();
    auto i               = index.global;
    auto updates_shape   = updates_t.get_shape();
    auto output_shape    = output_t.get_shape();
    const auto* data_ptr = data_t.data();
    auto* output_ptr     = output_t.data();

    if(i < output_shape.elements())
        output_ptr[i] = data_ptr[i];
    __syncthreads();

    if(i < updates_shape.elements())
    {
        const bool is_add       = s.is_add;
        const bool is_mul       = s.is_mul;
        const auto* indices_ptr = indices_t.data();
        const auto* updates_ptr = updates_t.data();

        auto indices_shape = indices_t.get_shape();
        auto k             = indices_shape.lens.back();
        auto q             = indices_shape.lens.size();

        auto updates_idx = updates_shape.multi(i);
        auto indices_idx = indices_shape.multi(0);
        for(std::size_t j = 0; j < q - 1; ++j)
            indices_idx[j] = updates_idx[j];

        auto* index_start = indices_ptr + indices_shape.index(indices_idx);
        auto out_idx      = output_shape.multi(0);
        for(std::size_t j = 0; j < k; ++j)
            out_idx[j] = index_start[j];

        for(std::size_t j = q - 1; j < updates_idx.size(); ++j)
            out_idx[j + k] = updates_idx[j];

        if(is_add)
            output_ptr[output_shape.index(out_idx)] += updates_ptr[i];
        else if(is_mul)
            output_ptr[output_shape.index(out_idx)] *= updates_ptr[i];
        else
            output_ptr[output_shape.index(out_idx)] = updates_ptr[i];
    }
}

} // namespace migraphx
#endif
