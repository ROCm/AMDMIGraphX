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
// shape of arg_2 is {768}
// b_arg2 = multibroadcast({batch_size, 128, 768}, arg2)
// sum_arg = add(arg1, b_arg2)
// rs_arg = reshape({batch_size, 128, 12, 64}, sum_arg)
// tr_arg = transpose([0, 2, 1, 3], rs_arg)
void add_transpose_arg0(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    const index_int block_size = 256;
    const index_int chunk_size = 768;
    const index_int chunks_per_block = 4;
    const index_int item_num = arg1.get_shape().elements();

    auto out_s = result.get_shape();
    auto out_lens = out_s.lens();
    auto in_lens = out_lens;
    in_lens[2] = out_lens[1];
    in_lens[1] = out_lens[2];
    shape arg_shape{out_s.type(), in_lens};
    auto original_out_stride = out_s.strides();
    auto out_stride = original_out_stride;
    out_stride[1] = original_out_stride[2];
    out_stride[2] = original_out_stride[1];
    shape ret_shape{out_s.type(), out_lens, out_stride};
    std::cout << "add_transpose_arg0, arg1_shape = " << arg1.get_shape() << std::endl;
    std::cout << "add_transpose_arg0, arg2_shape = " << arg2.get_shape() << std::endl;
    std::cout << "block_num = " << item_num / (chunks_per_block * 3 * block_size) << std::endl << std::endl;

    visit_all(arg1, arg2)([&](auto input1, auto input2) {
    hip_visit_all(result, ret_shape, arg_shape)([&](auto output, auto out_shape, auto in_shape) {
        auto *output_ptr = device_cast(output.data());
        auto *input1_ptr = device_cast(input1.data());
        auto *input2_ptr = device_cast(input2.data());

        launch(stream,
               item_num / (chunks_per_block * 3),
               block_size)([=](auto idx) __device__ {
            using type    = device_type<std::remove_cv_t<typename decltype(input1)::value_type>>;
            MIGRAPHX_DEVICE_SHARED type shared_input[chunk_size];

            idx.local_stride(chunk_size, [&](auto i) {
                shared_input[i] = input2_ptr[i];
            });
            __syncthreads();

            index_int blk_idx = idx.group;
            index_int start_idx = blk_idx * chunk_size * chunks_per_block;
            for (std::size_t c_no = 0; c_no < chunks_per_block; ++c_no)
            {
                index_int index = c_no * chunk_size + start_idx;
                idx.local_stride(chunk_size, [&](auto i) {
                    auto val = input1_ptr[index + i] + shared_input[i];
                    auto in_idx = in_shape.multi(index + i);
                    auto out_idx = in_idx;
                    output_ptr[out_shape.index(out_idx)] = val;
                });
                __syncthreads();
            }
        });
    });
    });
}

// the operator performed in this kernel is:
// shape of arg_2 is {768}
// b_arg2 = multibroadcast({batch_size, 128, 768}, arg2)
// sum_arg = add(arg1, b_arg2)
// rs_arg = reshape({batch_size, 128, 12, 64}, sum_arg)
// tr_arg = transpose([0, 2, 3, 1], rs_arg)
void add_transpose_arg1(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    const index_int block_size     = 256;
    const index_int chunk_size = 768;
    const index_int chunks_per_block = 4;
    const index_int item_num = arg1.get_shape().elements();
    auto out_s = result.get_shape();
    auto out_lens = out_s.lens();
    auto in_lens = out_lens;
    in_lens[3] = out_lens[2];
    in_lens[2] = out_lens[1];
    in_lens[1] = out_lens[3];
    shape arg_shape{out_s.type(), in_lens};
    auto original_out_stride = out_s.strides();
    auto out_stride = original_out_stride;
    out_stride[1] = original_out_stride[2];
    out_stride[2] = original_out_stride[3];
    out_stride[3] = original_out_stride[1];
    shape ret_shape{out_s.type(), out_lens, out_stride};
    std::cout << "add_transpose_arg1, arg1_shape = " << arg1.get_shape() << std::endl;
    std::cout << "add_transpose_arg1, arg2_shape = " << arg2.get_shape() << std::endl;
    std::cout << "block_num = " << item_num / (chunks_per_block * 3 * block_size) << std::endl << std::endl;

    visit_all(arg1, arg2)([&](auto input1, auto input2) {
    hip_visit_all(result, ret_shape, arg_shape)([&](auto output, auto out_shape, auto in_shape) {
        auto *output_ptr = device_cast(output.data());
        auto *input1_ptr = device_cast(input1.data());
        auto *input2_ptr = device_cast(input2.data());

        launch(stream,
               item_num / (chunks_per_block * 3),
               block_size)([=](auto idx) __device__ {
            using type    = device_type<std::remove_cv_t<typename decltype(input1)::value_type>>;
            MIGRAPHX_DEVICE_SHARED type shared_input[chunk_size];

            idx.local_stride(chunk_size, [&](auto i) {
                shared_input[i] = input2_ptr[i];
            });
            __syncthreads();

            index_int blk_idx = idx.group;
            index_int start_idx = blk_idx * chunk_size * chunks_per_block;
            for (std::size_t c_no = 0; c_no < chunks_per_block; ++c_no)
            {
                index_int index = c_no * chunk_size + start_idx;
                idx.local_stride(chunk_size, [&](auto i) {
                    auto val = input1_ptr[index + i] + shared_input[i];
                    auto in_idx = in_shape.multi(index + i);
                    auto out_idx = in_idx;
                    output_ptr[out_shape.index(out_idx)] = val;
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
