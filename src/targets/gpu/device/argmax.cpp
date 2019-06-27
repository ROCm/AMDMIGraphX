#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/argmax.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
inline __device__ void reduce_argmax(T* data_ptr,
                                     int64_t* index_ptr,
                                     std::size_t block_size,
                                     std::size_t thr_idx,
                                     std::size_t item_num,
                                     std::size_t max_index)
{
    while(true)
    {
        auto stride = (item_num + 1) / 2;
        auto size   = item_num / 2;
        for(std::size_t i = thr_idx; i < size; i += block_size)
        {
            if(data_ptr[i] < data_ptr[i + stride])
            {
                data_ptr[i]  = data_ptr[i + stride];
                index_ptr[i] = index_ptr[i + stride];
            }
        }
        __syncthreads();
        item_num = stride;

        if(item_num == 1)
            break;
    }

    if(thr_idx == 0)
    {
        if(data_ptr[max_index] < data_ptr[0])
        {
            data_ptr[max_index]  = data_ptr[0];
            index_ptr[max_index] = index_ptr[0];
        }
    }

    __syncthreads();
}

void argmax(hipStream_t stream, const argument& result, const argument& arg, int axis)
{
    auto arg_shape        = arg.get_shape();
    auto lens             = arg_shape.lens();
    auto batch_lens       = lens;
    size_t batch_item_num = lens[axis];
    batch_lens[axis]      = 1;
    migraphx::shape batch_shape{arg_shape.type(), batch_lens};

    hip_visit_all(arg, arg_shape, batch_shape)([&](auto input, auto arg_s, auto batch_s) {
        auto output = device_cast(result.get<int64_t>().data());
        // use one block for items in one batch.
        const size_t max_block_size = 1024;
        size_t block_size           = 1;
        while(block_size < max_block_size and block_size < batch_item_num)
        {
            block_size *= 2;
        }

        launch(stream, batch_shape.elements() * block_size, block_size)([=](auto idx) __device__ {
            size_t thr_idx = idx.local;
            size_t blk_idx = idx.group;
            using type     = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;

            auto batch_idx = batch_s.multi(blk_idx);
            auto data_idx  = batch_idx;
            MIGRAPHX_DEVICE_SHARED type lds_data[max_block_size + 1];
            MIGRAPHX_DEVICE_SHARED int64_t lds_index[max_block_size + 1];
            // load data to lds_data
            size_t round_item_num     = (batch_item_num + block_size - 1) / block_size * block_size;
            size_t remaining_item_num = batch_item_num;
            data_idx[axis] = 0;
            lds_data[max_block_size]  = input[arg_s.index(data_idx)];
            lds_index[max_block_size] = 0;
            for(size_t i = thr_idx; i < round_item_num; i += block_size)
            {
                if(i < batch_item_num)
                {
                    data_idx[axis]     = i;
                    lds_index[thr_idx] = i;
                    lds_data[thr_idx]  = input[arg_s.index(data_idx)];
                }
                __syncthreads();

                auto item_num = (remaining_item_num > block_size) ? block_size : remaining_item_num;
                reduce_argmax(lds_data, lds_index, block_size, thr_idx, item_num, max_block_size);

                remaining_item_num -= block_size;
            }

            if(thr_idx == 0)
            {
                output[batch_s.index(batch_idx)] = lds_index[max_block_size];
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
