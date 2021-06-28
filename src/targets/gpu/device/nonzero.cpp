#include "migraphx/gpu/device/visit.hpp"
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/float_equal.hpp>
#include <migraphx/gpu/device/shape.hpp>

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
    int nonzero_num;
    visit_all(result, arg_idx)([&](auto output, auto idx) {
        auto* idx_ptr = reinterpret_cast<int*>(device_cast(idx.data()));
        auto* out_ptr = device_cast(output.data());
        arg_data.visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            (void)hipMemset(idx_ptr, 0, sizeof(int));
            gs_launch(stream, 1, 1)([=](auto) __device__ {
                int index = 0;
                for(std::size_t i = 0; i < elem_num; ++i)
                    if(not float_equal(input_ptr[i], 0))
                    {
                        out_ptr[index++] = i;
                    }
                *idx_ptr = index;
            });
        });
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(&nonzero_num, idx_ptr, sizeof(int), hipMemcpyDeviceToHost);
    });

    result.visit([&](auto output) {
        auto* out_ptr = device_cast(output.data());
        hip_visit_all(arg_data.get_shape())([&](auto si) {
            gs_launch(stream, nonzero_num)([=](auto i) __device__ {
                auto index = si.multi(out_ptr[i]);
                for(std::size_t j = 0; j < index.size(); ++j)
                {
                    out_ptr[j * nonzero_num + i] = index[j];
                }
            });
        });
    });

    auto out_lens = result.get_shape().lens();
    out_lens[1]   = nonzero_num;
    shape out_s{result.get_shape().type(), out_lens};

    return {out_s, result.data()};
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
