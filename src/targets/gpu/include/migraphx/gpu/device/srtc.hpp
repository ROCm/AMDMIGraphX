#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SRTC_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SRTC_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template<int T0, int T1, int T2, int T3>
void slice_reshape_transpose(hipStream_t stream,
                        const argument& result,
                        const argument& arg,
                        int slice_start)
{
    const index_int block_size       = 256;
    const index_int chunk_size       = 768;
    const index_int chunks_per_block = 4;
    auto out_s                       = result.get_shape();
    const index_int item_num         = out_s.elements();
    auto out_lens                    = out_s.lens();
    slice_start /= 64;

    auto arg_s    = arg.get_shape();
    auto arg_lens = arg_s.lens();
    arg_lens.resize(out_lens.size());
    arg_lens[3] = 64;
    arg_lens[2] = arg_lens[2] / arg_lens[3];
    shape arg_shape{arg_s.type(), arg_lens};
    auto arg_stride = arg_shape.strides();

    auto in_lens   = arg_lens;
    in_lens[0]     = arg_lens[T0];
    in_lens[1]     = arg_lens[T1];
    in_lens[2]     = arg_lens[T2];
    in_lens[3]     = arg_lens[T3];

    auto in_stride = arg_stride;
    in_stride[0]   = arg_stride[T0];
    in_stride[1]   = arg_stride[T1];
    in_stride[2]   = arg_stride[T2];
    in_stride[3]   = arg_stride[T3];

    shape in_s{arg_s.type(), in_lens, in_stride};

    arg.visit([&](auto input) {
        hip_visit_all(result, out_s, in_s)([&](auto output, auto out_shape, auto in_shape) {
            auto* output_ptr = device_cast(output.data());
            auto* input_ptr  = device_cast(input.data());

            launch(stream, item_num / (chunks_per_block * 3), block_size)([=](auto idx) __device__ {

                index_int blk_idx   = idx.group;
                index_int start_idx = blk_idx * chunk_size * chunks_per_block;
                for(std::size_t c_no = 0; c_no < chunks_per_block; ++c_no)
                {
                    index_int index = c_no * chunk_size + start_idx;
                    idx.local_stride(chunk_size, [&](auto i) {
                        auto out_idx = out_shape.multi(index + i);
                        auto in_idx  = out_idx;
                        in_idx[1] += slice_start;
                        int j                 = in_shape.index(in_idx);
                        output_ptr[index + i] = input_ptr[j];
                    });
                    __syncthreads();
                }
            });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
