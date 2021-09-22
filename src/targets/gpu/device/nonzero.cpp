#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/float_equal.hpp>
#include <migraphx/gpu/device/scan.hpp>
#include <migraphx/gpu/device/fill.hpp>
#include <migraphx/gpu/device/reduce_ops.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream,
                 const argument& result,
                 const argument& arg_idx,
                 const argument& arg_data)
{
    auto s        = arg_data.get_shape();
    auto elem_num = s.elements();
    auto n_dim    = s.lens().size();

    // call the prefix_sum function to do a prefix_sum to compute
    // index in the output. Only 1 block can be used since we have
    // only one prefix sum
    const index_int block_size = 256;
    arg_data.visit([&](auto input) {
        const auto* in_ptr = device_cast(input.data());
        auto* ptr          = arg_idx.cast<int64_t>();
        gs_launch(stream, block_size, block_size)([=](auto, auto idx) __device__ {
            block_scan<block_size>(idx,
                                   sum{},
                                   0,
                                   elem_num,
                                   [&](auto j) { return (float_equal(in_ptr[j], 0)) ? 0 : 1; },
                                   [&](auto j, auto x) { ptr[j] = x; });
        });
    });

    auto* idx     = arg_idx.cast<int64_t>();
    auto* out_ptr = result.cast<int64_t>();
    hip_visit_all(s)([&](auto si) {
        gs_launch(stream, elem_num)([=](auto i) __device__ {
            auto nz_elem_num = idx[elem_num - 1];
            if(i >= nz_elem_num)
            {
                for(std::size_t j = 0; j < n_dim; ++j)
                {
                    out_ptr[j * elem_num + i] = 0;
                }
            }

            if((i == 0 and idx[i] == 0) or (i > 0 and idx[i] == idx[i - 1]))
                return;

            auto out_idx = idx[i] - 1;
            auto index   = si.multi(i);
            for(std::size_t j = 0; j < index.size(); ++j)
            {
                out_ptr[j * elem_num + out_idx] = index[j];
            }
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
