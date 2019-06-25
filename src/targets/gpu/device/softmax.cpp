#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce_opers.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void softmax(hipStream_t stream, const argument& result, const argument& arg, int axis)
{
    auto lens             = result.get_shape().lens();
    auto batch_lens       = lens;
    size_t batch_item_num = lens[axis];
    batch_lens[axis]      = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    visit_all(result, arg)([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());
        visit_tensor_size(batch_shape.lens().size(), [&](auto n_dim) {
            hip_tensor_descriptor<n_dim> desc_batch(batch_shape);
            hip_tensor_descriptor<n_dim> desc_data(result.get_shape());

            // use one block for items in one batch.
            const size_t max_block_size = 1024;
            size_t block_size           = 1;
            while(block_size < max_block_size and block_size < batch_item_num)
            {
                block_size *= 2;
            }

            launch(
                stream, batch_shape.elements() * block_size, block_size)([=](auto idx) __device__ {
                size_t thr_idx = idx.local;
                size_t blk_idx = idx.group;
                using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;

                MIGRAPHX_DEVICE_SHARED type lds_data[max_block_size + 1];
                auto batch_idx = desc_batch.multi(blk_idx);
                auto data_idx  = batch_idx;
                // load data to lds and compute the batch max
                size_t remaining_item_num = batch_item_num;
                size_t round_item_num = (batch_item_num + block_size - 1) / block_size * block_size;
                lds_data[block_size]  = input_ptr[0];
                for(size_t i = thr_idx; i < round_item_num; i += block_size)
                {
                    if(i < batch_item_num)
                    {
                        data_idx[axis]    = i;
                        lds_data[thr_idx] = input_ptr[desc_data.linear(data_idx)];
                    }

                    __syncthreads();

                    auto item_num =
                        (remaining_item_num > block_size) ? block_size : remaining_item_num;
                    reduce_max<type>(lds_data, block_size, thr_idx, item_num);

                    remaining_item_num -= block_size;
                }

                auto batch_max = lds_data[block_size];
                __syncthreads();

                lds_data[block_size] = 0;
                remaining_item_num   = batch_item_num;
                for(size_t i = thr_idx; i < round_item_num; i += block_size)
                {
                    if(i < batch_item_num)
                    {
                        data_idx[axis]    = i;
                        lds_data[thr_idx] = input_ptr[desc_data.linear(data_idx)] - batch_max;
                        lds_data[thr_idx] = ::exp(to_hip_type(lds_data[thr_idx]));
                    }

                    __syncthreads();

                    auto item_num =
                        (remaining_item_num > block_size) ? block_size : remaining_item_num;
                    reduce_sum<type>(lds_data, block_size, thr_idx, item_num);

                    remaining_item_num -= block_size;
                }
                auto batch_sum = lds_data[block_size];

                for(size_t i = thr_idx; i < batch_item_num; i += block_size)
                {
                    data_idx[axis]    = i;
                    size_t index      = desc_data.linear(data_idx);
                    auto val          = input_ptr[index] - batch_max;
                    output_ptr[index] = ::exp(to_hip_type(val)) / batch_sum;
                }
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
