#include <migraphx/gpu/device/prefix_scan_sum.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/tensor_view.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void prefix_scan_sum(hipStream_t stream,
                     const argument& result,
                     const argument& arg,
                     int32_t axis,
                     bool exclusive,
                     bool reverse)
{
    const index_int block_size = 256;
    const index_int n          = arg.get_shape().lens()[axis];
    auto rlens                 = result.get_shape().lens();
    rlens[axis]                = 1;

    auto s          = result.get_shape();
    auto bshape = shape{s.type(), rlens, s.strides()};
    auto stride = s.strides()[axis];
    auto len = s.lens()[axis];

    hip_visit_all(result,
                  arg,
                  result.get_shape().with_lens(rlens),
                  bshape)([=](auto output, auto input, auto rshape, auto batch) {
        
        if (exclusive)
        {   
            gs_launch(stream, rshape.elements()) ([=](auto i) __device__ {
                auto input_slice = input.begin() + batch.index(i);
                auto output_slice = output.begin() + batch.index(i);
                for (std::size_t j = 0; j < (len - 1) * stride; j += stride)
                {
                    auto out_ind = reverse ? j : (len * stride) - j - stride;
                    auto in_ind = reverse ? j + stride : (len * stride) - j - (2 * stride);
                    output_slice[out_ind] = input_slice[in_ind];
                }
                auto out_ind = reverse ? (len - 1) * stride : 0;
                output_slice[out_ind] = 0; 
            });
        } 
        gs_launch(
            stream, rshape.elements() * block_size, block_size)([=](auto i, auto idx) __device__ {
            const auto ridx  = rshape.multi(i / block_size);
            auto compute_idx = [&](auto j) {
                auto k  = ridx;
                k[axis] = j;
                return k;
            };
            if(reverse)
            {
                reverse_block_scan<block_size>(idx,
                                               sum{},
                                               0,
                                               n,
                                               [&](auto j) { return exclusive ? output[compute_idx(j)] : input[compute_idx(j)]; },
                                               [&](auto j, auto x) { output[compute_idx(j)] = x; });
            }
            else
            {
                block_scan<block_size>(idx,
                                       sum{},
                                       0,
                                       n,
                                       [&](auto j) { return exclusive ? output[compute_idx(j)] : input[compute_idx(j)]; },
                                       [&](auto j, auto x) { output[compute_idx(j)] = x; });
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
