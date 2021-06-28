#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/nonzero.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/float_equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream, const argument& result, const argument& arg_idx, const argument& arg_data)
{
    auto elem_num = arg_data.get_shape().elements();
    int nonzero_num;
    arg_idx.visit([&](auto idx) {
        auto* idx_ptr = reinterpret_cast<int*>(device_cast(idx.data()));
        arg_data.visit([&](auto input) {
            const auto* input_ptr = device_cast(input.data());
            (void)hipMemset(idx_ptr, 0, sizeof(int));
            gs_launch(stream, elem_num)([=](auto i) __device__ {
                if(not float_equal(input_ptr[i], 0))
                {
                    atomicAdd(idx_ptr, 1);
                }
            });
        });
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(&nonzero_num, idx_ptr, sizeof(int), hipMemcpyDeviceToHost);
    });

    result.visit([&](auto output) {
        hip_visit_all(arg_data, arg_data.get_shape())([&](auto input, auto si) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr      = device_cast(output.data());
            gs_launch(stream, 1, 1)([=](auto) __device__ {
                // this process have to be serial, so only use the first thread
                std::size_t out_idx = 0;
                for(std::size_t i = 0; i < elem_num; ++i)
                    if(not float_equal(input_ptr[i], 0))
                    {
                        auto idx = si.multi(i);
                        for(std::size_t j = 0; j < idx.size(); ++j)
                            output_ptr[j * nonzero_num + out_idx] = idx[j];
                        ++out_idx;
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
