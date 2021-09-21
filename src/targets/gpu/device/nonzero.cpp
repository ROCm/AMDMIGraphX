#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/float_equal.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/fill.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <hip/hip_runtime.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream,
                 const argument& result,
                 const argument& arg_idx,
                 const argument& arg_data)
{
    auto elem_num = arg_data.get_shape().elements();
    // check nonzero elements
    // arg_data.visit([&](auto input) {
    //     const auto* in_ptr = device_cast(input.data());
    //     auto* idx          = arg_idx.cast<int64_t>();
    //     gs_launch(stream, elem_num)(
    //         [=](auto i) __device__ { idx[i] = (float_equal(in_ptr[i], 0)) ? 0 : 1; });
    // });

    // set all output values to 0
    // fill(stream, result, 0);

    // call the prefix_sum function to do a prefix_sum to compute
    // index in the output. Only 1 block can be used since we have
    // only one prefix sum
    const index_int block_size = 256;
    auto out_elem_num = result.get_shape().elements();
    hip_visit_all(arg_data, arg_data.get_shape())([&](auto input, auto si) {
        auto* out_ptr = result.cast<int64_t>();
        const auto* data = device_cast(input.data());
        // arg_idx.visit([&](auto in_idx) {
        //     auto* ptr = device_cast(in_idx.data());
            gs_launch(stream, block_size, block_size)([=](auto, auto idx) __device__ {
                idx.local_stride(out_elem_num, [&](auto j) {
                    out_ptr[j] = 0;
                });

                block_scan<block_size>(idx,
                                    sum{},
                                    0,
                                    elem_num,
                                    [&](auto j) { return float_equal(data[j], 0) ? 0 : 1; },
                                    [&](auto, auto x) {
                                        auto out_idx = x - 1;
                                        auto index = si.multi(out_idx);
                                        for(std::size_t k = 0; k < index.size(); ++k)
                                        {
                                            // output after nz_elem_num should be set to 0
                                            out_ptr[k * elem_num + out_idx] = index[k];
                                        }
                                    });
            });
        // });
    });

    // result.visit([&](auto output) {
    //     auto* idx     = arg_idx.cast<int64_t>();
    //     auto* out_ptr = device_cast(output.data());
    //     hip_visit_all(arg_data.get_shape())([&](auto si) {
    //         gs_launch(stream, elem_num)([=](auto i) __device__ {
    //             // non-zero elem num
    //             int64_t nz_elem_num = idx[elem_num - 1];
    //             auto out_idx = idx[i] - 1;
    //             auto index   = si.multi(out_idx);
    //             for(std::size_t j = 0; j < index.size(); ++j)
    //             {
    //                 // output after nz_elem_num should be set to 0
    //                 if(i >= nz_elem_num)
    //                 {
    //                     out_ptr[j * elem_num + i] = 0;
    //                 }
    //                 out_ptr[j * elem_num + out_idx] = index[j];
    //             }
    //         });
    //     });
    // });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
