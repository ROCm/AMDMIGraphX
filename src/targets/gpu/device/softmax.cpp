#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/softmax.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

#define MAX_BATCH_ITEM_NUM 1024

void softmax(hipStream_t stream, const argument& result, const argument& arg, int64_t axis)
{
    auto batch_lens          = result.get_shape().lens();
    index_int batch_item_num = batch_lens[axis];
    assert(batch_item_num <= MAX_BATCH_ITEM_NUM);
    batch_lens[axis]         = 1;
    migraphx::shape batch_shape{result.get_shape().type(), batch_lens};

    hip_visit_all(result, arg, batch_shape)([&](auto output, auto input, auto batch) {
        const index_int max_block_size = 256;
        const index_int block_size     = compute_block_size(batch_item_num, max_block_size);
        gs_launch(stream,
                  batch_shape.elements() * block_size,
                  block_size)([=](auto i, auto idx) __device__ {
            auto data_idx = batch.multi(i / block_size);
            using type    = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
            type init     = lowest();
            MIGRAPHX_DEVICE_SHARED type sinput_data[MAX_BATCH_ITEM_NUM];
            idx.local_stride(batch_item_num, [&](auto j) {
                data_idx[axis] = j;
                sinput_data[j] = input[data_idx];
            });
            __syncthreads();

            auto batch_max = block_reduce<max_block_size>(
                idx, max{}, init, batch_item_num, [&](auto j) __device__ {
                    return sinput_data[j];
                });

            auto batch_sum =
                block_reduce<max_block_size>(idx, sum{}, 0, batch_item_num, [&](auto j) __device__ {
                    auto val = sinput_data[j] - batch_max;
                    sinput_data[j] = val;
                    return ::exp(to_hip_type(val));
                });

            idx.local_stride(batch_item_num, [&](auto j) __device__ {
                data_idx[axis] = j;
                output[data_idx] = ::exp(to_hip_type(sinput_data[j])) / batch_sum;
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
