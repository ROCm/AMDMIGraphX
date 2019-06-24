#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/argmin.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument argmax(hipStream_t stream, const argument& result, const argument& arg, int axis)
{
    auto lens        = arg.get_shape().lens();
    auto batch_lens  = lens;
    size_t n_dims    = lens[axis];
    batch_lens[axis] = 1;
    migraphx::shape batch_shape{shape::float_type, batch_lens};

    visit_all(result, arg)([&](auto output, auto input) {
        const auto* input_ptr = device_cast(input.data());
        auto* output_ptr      = device_cast(output.data());
        visit_tensor_size(batch_shape.lens().size(), [&](auto n_dim) {
            hip_tensor_descriptor<n_dim> desc_batch(batch_shape);
            hip_tensor_descriptor<n_dim> desc_data(arg.get_shape());

            // each block is for one batch
            const size_t block_size = 1024;
            launch(
                stream, batch_shape.elements() * block_size, block_size)([=](auto idx) __device__ {
                size_t thr_idx = idx.local;
                size_t blk_idx = idx.group;
                using type = device_type<std::remove_cv_t<typename decltype(output)::value_type>>;

                auto batch_idx = desc_batch.multi(blk_idx);
                auto data_idx  = batch_idx;
                MIGRAPHX_DEVICE_SHARED type lds_data[block_size];
                MIGRAPHX_DEVICE_SHARED int64_t lds_index[block_size];
                // load data to lds_data
                size_t item_num = n_dims;
                for(size_t i = thr_idx; i < n_dims; i += block_size)
                {
                    data_idx[axis]     = i;
                    lds_index[thr_idx] = i;
                    lds_data[thr_idx]  = input_ptr[desc_data.linear(data_idx)];
                    __syncthreads();

                    auto size   = (item_num > block_size) ? block_size : item_num;
                    auto stride = (size + 1) / 2;
                    while(true)
                    {
                        if(thr_idx + stride < size and
                           lds_data[thr_idx] > lds_data[thr_idx + stride])
                        {
                            lds_data[thr_idx]  = lds_data[thr_idx + stride];
                            lds_index[thr_idx] = lds_index[thr_idx + stride];
                        }

                        __syncthreads();
                        size   = stride;
                        stride = (stride + 1) / 2;

                        if(size == 1)
                            break;
                    }

                    if(thr_idx == 0)
                    {
                        output_ptr[blk_idx] = lds_index[0];
                    }

                    item_num -= block_size;
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
