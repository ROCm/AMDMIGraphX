#ifndef MIGRAPHX_GUARD_OPERATORS_GATHERND_HPP
#define MIGRAPHX_GUARD_OPERATORS_GATHERND_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct gathernd
{
    int batch_dims = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.batch_dims, "batch_dims"));
    }

    std::string name() const { return "gathernd"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).standard();
        auto r = inputs.front().lens().size();
        auto q = inputs.back().lens().size();
        auto k = inputs.back().lens().back();
        auto indices_lens_iter = inputs.back().lens().begin();
        auto output_lens_size = q + r - k - batch_dims - 1;
        std::vector<std::size_t> output_lens(output_lens_size);
        std::copy(indices_lens_iter, indices_lens_iter + (q - 1), output_lens.begin());
        if (k < r - batch_dims) {
            auto data_lens = inputs.front().lens();
            std::copy(data_lens.begin() + batch_dims + k, data_lens.end(), output_lens.begin() + q - 1);
        }
        shape output_shape{inputs.front().type(), output_lens};
        return output_shape;
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0])([&](auto output, auto data) {
            args[1].visit([&](auto indices) {
                auto indices_shape = indices.get_shape();
                auto indices_shape_lens = indices_shape.lens();
                auto data_shape = data.get_shape();
                auto data_shape_lens = data_shape.lens();
                auto k = indices_shape.lens().back();

                // Prepare Compute Params
                const auto num_slice_dims = k;
                std::size_t num_slices = std::accumulate(indices_shape_lens.begin(), indices_shape_lens.end() - 1, 1, std::multiplies<std::size_t>());
                std::size_t slice_size = std::accumulate(data_shape_lens.begin() + k + batch_dims, data_shape_lens.end(), 1, std::multiplies<std::size_t>());
                const std::size_t num_batches = std::accumulate(data_shape_lens.begin(), data_shape_lens.begin() + batch_dims, 1, std::multiplies<std::size_t>());
                const std::size_t data_batch_stride = std::accumulate(data_shape_lens.begin() + batch_dims, data_shape_lens.end(), 1, std::multiplies<std::size_t>());
                const auto num_slices_per_batch = num_slices / num_batches;

                std::vector<std::size_t> sizes_from_slice_dims(num_slice_dims);
                {
                    auto running_product = slice_size;
                    for (std::size_t i = 0; i < num_slice_dims; ++i) {
                        sizes_from_slice_dims[num_slice_dims - 1 - i] = running_product;
                        running_product *= data_shape_lens[batch_dims + num_slice_dims - 1 - i];
                    }
                }

                std::vector<std::size_t> input_slice_offsets(num_slices);

                // Compute Slice Offset
                par_for(num_slices, [&](const auto i) {
                    const std::size_t batch_idx = i / num_slices_per_batch;
                    const std::size_t base_offset = batch_idx * data_batch_stride;

                    auto* slice_indices = indices.data() + (i * num_slice_dims);
                    std::size_t relative_slice_offset = 0;
                    for (size_t dim_idx = 0; dim_idx < num_slice_dims; ++dim_idx) {
                        int64_t index = static_cast<int64_t>(slice_indices[dim_idx]);
                        const std::size_t input_dim_idx = batch_dims + dim_idx;
                        //CUDA_KERNEL_ASSERT(index >= -input_dims[input_dim_idx] && index < input_dims[input_dim_idx]);
                        if (index < 0) index += data_shape_lens[input_dim_idx];

                        relative_slice_offset += index * sizes_from_slice_dims[dim_idx];
                    }

                    input_slice_offsets[i] = base_offset + relative_slice_offset;
                });

                // Compute
                par_for(num_slices * slice_size, [&](const auto i) {
                    auto slice_offset = input_slice_offsets[i / slice_size];
                    output[i] = data[slice_offset + i % slice_size];
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
