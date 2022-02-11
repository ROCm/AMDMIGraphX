#ifndef MIGRAPHX_GUARD_KERNELS_GATHERND_HPP
#define MIGRAPHX_GUARD_KERNELS_GATHERND_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>

namespace migraphx {

template <class T>
struct gathernd_settings
{
    T batch_dims{};
};

template <class... Ts>
constexpr gathernd_settings<Ts...> make_gathernd_settings(Ts... xs)
{
    return {xs...};
}

template <class T, class U, class V, class Settings>
__device__ void gathernd(const T& data_t, const U& indices_t, const V& output_t, Settings s)
{
    auto ind           = make_index();
    auto i             = ind.global;
    auto batch_dims    = s.batch_dims;
    auto output_shape  = output_t.get_shape();
    auto indices_shape = indices_t.get_shape();
    auto data_shape    = data_t.get_shape();

    auto indices_shape_lens = indices_shape.lens;
    auto data_shape_lens    = data_shape.lens;
    auto num_slice_dims     = indices_shape_lens.back();
    std::size_t num_slices  = accumulate(indices_shape_lens.begin(),
                                        indices_shape_lens.end() - 1,
                                        1,
                                        std::multiplies<std::size_t>());
    std::size_t slice_size  = accumulate(data_shape_lens.begin() + num_slice_dims + batch_dims,
                                        data_shape_lens.end(),
                                        1,
                                        std::multiplies<std::size_t>());
    const std::size_t num_batches       = accumulate(data_shape_lens.begin(),
                                               data_shape_lens.begin() + batch_dims,
                                               1,
                                               std::multiplies<std::size_t>());
    const std::size_t data_batch_stride = accumulate(data_shape_lens.begin() + batch_dims,
                                                     data_shape_lens.end(),
                                                     1,
                                                     std::multiplies<std::size_t>());
    const auto num_slices_per_batch     = num_slices / num_batches;

    if(i < output_shape.elements())
    {
        const auto* indices_ptr       = indices_t.data();
        const std::size_t j           = i / slice_size;
        const std::size_t batch_idx   = j / num_slices_per_batch;
        const std::size_t base_offset = batch_idx * data_batch_stride;

        auto* slice_indices               = indices_ptr + (j * num_slice_dims);
        std::size_t relative_slice_offset = 0;
        for(std::size_t idx = 0; idx < num_slice_dims; ++idx)
        {
            int64_t index                   = slice_indices[idx];
            const std::size_t input_dim_idx = batch_dims + idx;
            assert(index >= -static_cast<int64_t>(data_shape_lens[input_dim_idx]) and
                   index < static_cast<int64_t>(data_shape_lens[input_dim_idx]));
            if(index < 0)
                index += data_shape_lens[input_dim_idx];
            std::size_t size_from_slice_dims =
                accumulate(data_shape_lens.begin() + batch_dims + idx + 1,
                           data_shape_lens.begin() + batch_dims + num_slice_dims,
                           slice_size,
                           std::multiplies<std::size_t>());
            relative_slice_offset += index * size_from_slice_dims;
        }

        auto slice_offset = base_offset + relative_slice_offset;
        output_t[i]       = data_t[slice_offset + i % slice_size];
    }
}

} // namespace migraphx
#endif
