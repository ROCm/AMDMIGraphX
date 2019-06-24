#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument softmax(hipStream_t stream,
                 const migraphx::shape& output_shape,
                 std::vector<migraphx::argument> args,
                 int axis)
{
    auto lens        = output_shape.lens();
    auto batch_lens  = lens;
    size_t n_dims    = lens[axis];
    batch_lens[axis] = 1;
    migraphx::shape batch_shape{output_shape.type(), batch_lens};

    visit_all(args.back(), args.front())([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());
        visit_tensor_size(batch_shape.lens().size(), [&](auto n_dim) {
            hip_tensor_descriptor<n_dim> desc_batch(batch_shape);
            hip_tensor_descriptor<n_dim> desc_data(output_shape);

            // use one block for items in one batch.
            const size_t max_block_size = 1024;
            size_t block_size = 1;
            while (block_size < max_block_size and block_size < n_dims)
            {
                block_size *= 2;
            }

            launch(
                stream, batch_shape.elements() * block_size, block_size)([=](auto idx) __device__ {
                size_t thr_idx = idx.local;
                size_t blk_idx = idx.group;
                using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;

                // all data can be loaded to the lds once, so all operations are
                // done in lds
                MIGRAPHX_DEVICE_SHARED type lds_data[max_block_size + 2];
                auto batch_idx = desc_batch.multi(blk_idx);
                auto data_idx  = batch_idx;
                // load data to lds and compute the batch max
                size_t item_num      = n_dims;
                size_t thread_num = (n_dims + block_size - 1) / block_size * block_size;
                lds_data[block_size] = input_ptr[0];
                lds_data[block_size + 1]    = 0;
                for(size_t i = thr_idx; i < thread_num; i += block_size)
                {
                    if (i < n_dims)
                    {
                        data_idx[axis] = i;
                        lds_data[thr_idx]    = input_ptr[desc_data.linear(data_idx)];
                    }

                    __syncthreads();

                    auto size   = (item_num > block_size) ? block_size : item_num;
                    auto stride = (size + 1) / 2;
                    while(true)
                    {
                        if(thr_idx + stride < size)
                        {
                            lds_data[thr_idx] = ::max(to_hip_type(lds_data[thr_idx]),
                                                      to_hip_type(lds_data[thr_idx + stride]));
                        }
                        __syncthreads();
                        size   = stride;
                        stride = (stride + 1) / 2;

                        if(size == 1)
                            break;
                    }

                    if(thr_idx == 0)
                    {
                        lds_data[block_size] = (lds_data[0] < lds_data[block_size])
                                                   ? lds_data[block_size]
                                                   : lds_data[0];
                    }
                    __syncthreads();

                    item_num -= block_size;
                }

                item_num                 = n_dims;
                for(size_t i = thr_idx; i < thread_num; i += block_size)
                {
                    if (i < n_dims)
                    {
                        data_idx[axis] = i;
                        lds_data[thr_idx]    = input_ptr[desc_data.linear(data_idx)] - lds_data[block_size];
                        lds_data[thr_idx]    = ::exp(to_hip_type(lds_data[thr_idx]));
                    }

                    __syncthreads();

                    auto size   = (item_num > block_size) ? block_size : item_num;
                    auto stride = (size + 1) / 2;
                    while(true)
                    {
                        if(thr_idx + stride < size)
                        {
                            lds_data[thr_idx] += lds_data[thr_idx + stride];
                        }
                        __syncthreads();
                        size   = stride;
                        stride = (stride + 1) / 2;
                        if(size == 1)
                            break;
                    }

                    if(thr_idx == 0)
                    {
                        lds_data[block_size + 1] += lds_data[0];
                    }
                    __syncthreads();

                    item_num -= block_size;
                }

                for(size_t i = thr_idx; i < n_dims; i += block_size)
                {
                    data_idx[axis]    = i;
                    size_t index      = desc_data.linear(data_idx);
                    auto val          = input_ptr[index] - lds_data[block_size];
                    output_ptr[index] = ::exp(to_hip_type(val)) / lds_data[block_size + 1];
                }
            });
        });
    });

    return args.back();
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
