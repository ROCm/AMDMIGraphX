#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/logsoftmax.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument logsoftmax(hipStream_t stream,
                    const migraphx::shape& output_shape,
                    std::vector<migraphx::argument> args,
                    int axis)
{

    auto lens         = output_shape.lens();
    auto num_in_batch = lens[axis];
    auto batch_lens   = lens;
    batch_lens[axis]  = 1;
    migraphx::shape batch_shape{output_shape.type(), batch_lens};

    visit_all(args.back(), args.front())([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());
        visit_tensor_size(batch_shape.lens().size(), [&](auto n_dim) {
            hip_tensor_descriptor<n_dim> desc_batch(batch_shape);
            hip_tensor_descriptor<n_dim> desc_data(output_shape);

            // use one block for items in one batch.
            // opt 1, load all data to lds then use the same approach as
            // the current optimization
            const size_t block_size = 1024;
            launch(
                stream, batch_shape.elements() * block_size, block_size)([=](auto idx) __device__ {
                size_t thr_idx = idx.local;
                size_t blk_idx = idx.group;
                using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;

                // all data can be loaded to the lds once, so all operations are
                // done in lds
                MIGRAPHX_DEVICE_SHARED type lds_data[block_size + 2];
                auto batch_idx = desc_batch.multi(blk_idx);
                auto data_idx  = batch_idx;
                // load data to lds and compute the batch max
                size_t item_num      = num_in_batch;
                lds_data[block_size] = input_ptr[0];
                for(size_t i = thr_idx; i < num_in_batch; i += block_size)
                {
                    data_idx[axis] = i;
                    lds_data[i]    = input_ptr[desc_data.linear(data_idx)];

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

                const size_t block_size1 = block_size + 1;
                lds_data[block_size1]    = 0;
                item_num                 = num_in_batch;
                for(size_t i = thr_idx; i < num_in_batch; i += block_size)
                {
                    data_idx[axis] = i;
                    lds_data[i]    = input_ptr[desc_data.linear(data_idx)] - lds_data[block_size];
                    lds_data[i]    = ::exp(to_hip_type(lds_data[i]));

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
                        lds_data[block_size1] += lds_data[0];
                    }
                    __syncthreads();

                    item_num -= block_size;
                }

                auto log_batch_sum =
                    ::log(to_hip_type(lds_data[block_size1])) + lds_data[block_size];
                item_num = num_in_batch;
                for(size_t i = thr_idx; i < num_in_batch; i += block_size)
                {
                    data_idx[axis]    = i;
                    size_t index      = desc_data.linear(data_idx);
                    output_ptr[index] = input_ptr[index] - log_batch_sum;
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
