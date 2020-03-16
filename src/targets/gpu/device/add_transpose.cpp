#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/dfor.hpp>
#include <migraphx/gpu/device/add_transpose.hpp>
#include <migraphx/gpu/device/reduce.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// the operator performed in this kernel is:
// shape of arg is {1, 128, 2304}
// slice {1536, 2304) of arg to generate shape of {1, 128, 768}
// reshape to shape of ({batch_size, 128, 12, 64}, sum_arg)
// transpose to shape of ([0, 2, 1, 3], rs_arg)
void add_transpose_arg0(hipStream_t stream, const argument& result, const argument& arg, int slice_start)
{
    const index_int block_size = 256;
    const index_int chunk_size = 768;
    const index_int chunks_per_block = 4;
    auto out_s = result.get_shape();
    const index_int item_num = out_s.elements();
    auto out_lens = out_s.lens();
    slice_start /= 64;

    auto arg_s = arg.get_shape();
    auto arg_lens = arg_s.lens();
    arg_lens.resize(out_lens.size());
    arg_lens[3] = 64;
    arg_lens[2] = arg_lens[2] / arg_lens[3];
    shape arg_shape{arg_s.type(), arg_lens};
    auto arg_stride = arg_shape.strides();

    auto in_lens = arg_lens;
    in_lens[1] = arg_lens[2];
    in_lens[2] = arg_lens[1];
    auto in_stride = arg_stride;
    in_stride[1] = arg_stride[2];
    in_stride[2] = arg_stride[1];
    shape in_s{arg_s.type(), in_lens, in_stride};

    arg.visit([&](auto input) {
    hip_visit_all(result, out_s, in_s)([&](auto output, auto out_shape, auto in_shape) {
        auto *output_ptr = device_cast(output.data());
        auto *input_ptr = device_cast(input.data());

        launch(stream,
               item_num / (chunks_per_block * 3),
               block_size)([=](auto idx) __device__ {

            index_int blk_idx = idx.group;
            index_int start_idx = blk_idx * chunk_size * chunks_per_block;
            for (std::size_t c_no = 0; c_no < chunks_per_block; ++c_no)
            {
                index_int index = c_no * chunk_size + start_idx;
                idx.local_stride(chunk_size, [&](auto i) {
                    auto out_idx = out_shape.multi(index + i);
                    auto in_idx = out_idx;
                    in_idx[1] += slice_start;
                    int j = in_shape.index(in_idx);
                    output_ptr[index + i] = input_ptr[j];
                });
                __syncthreads();
            }
        });
    });
    });
}

// the operator performed in this kernel is:
// shape of arg is {1, 128, 2304}
// slice {768, 1536) of arg to generate shape of {1, 128, 768}
// reshape to shape of ({batch_size, 128, 12, 64}, sum_arg)
// transpose to shape of ([0, 2, 3, 1], rs_arg)
void add_transpose_arg1(hipStream_t stream, const argument& result, const argument& arg, int slice_start)
{
    const index_int block_size = 256;
    const index_int chunk_size = 768;
    const index_int chunks_per_block = 4;
    const index_int block_iter = chunk_size / block_size;
    auto out_s = result.get_shape();
    const index_int item_num = out_s.elements();
    auto out_lens = out_s.lens();
    slice_start /= 64;

    auto arg_s = arg.get_shape();
    auto arg_lens = arg_s.lens();
    arg_lens.resize(out_lens.size());
    arg_lens[3] = 64;
    arg_lens[2] = arg_lens[2] / arg_lens[3];
    shape arg_shape{arg_s.type(), arg_lens};
    auto arg_stride = arg_shape.strides();

    auto in_lens = arg_lens;
    in_lens[1] = arg_lens[2];
    in_lens[2] = arg_lens[3];
    in_lens[3] = arg_lens[1];
    auto in_stride = arg_stride;
    in_stride[1] = arg_stride[2];
    in_stride[2] = arg_stride[3];
    in_stride[3] = arg_stride[1];
    shape in_s{arg_s.type(), in_lens, in_stride};

    arg.visit([&](auto input) {
    hip_visit_all(result, out_s, in_s)([&](auto output, auto out_shape, auto in_shape) {
        auto *output_ptr = device_cast(output.data());
        auto *input_ptr = device_cast(input.data());

        launch(stream,
               item_num / (chunks_per_block * block_iter),
               block_size)([=](auto idx) __device__ {

            index_int blk_idx = idx.group;
            index_int start_idx = blk_idx * chunk_size * chunks_per_block;
            for (std::size_t c_no = 0; c_no < chunks_per_block; ++c_no)
            {
                index_int index = c_no * chunk_size + start_idx;
                idx.local_stride(chunk_size, [&](auto i) {
                    auto out_idx = out_shape.multi(index + i);
                    auto in_idx = out_idx;
                    in_idx[1] += slice_start;
                    int j = in_shape.index(in_idx);
                    output_ptr[index + i] = input_ptr[j];
                });
            }
        });
    });
    });
}


} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
