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

argument nonzero(hipStream_t stream, argument result, argument arg)
{
    auto elem_num = arg.get_shape().elements();
    int* nonzero_num;
    (void)hipMalloc(&nonzero_num, sizeof(int));
    (void)hipMemset(nonzero_num, 0, sizeof(int));
    arg.visit([&](auto input) {
        const auto* input_ptr = device_cast(input.data());
        gs_launch(stream, elem_num)([=](auto i) __device__ {
            if (not float_equal(input_ptr[i], 0))
            {
                atomicAdd(nonzero_num, 1);
            }
        });
    });

    result.visit([&](auto output) {
        hip_visit_all(arg, arg.get_shape())([&](auto input, auto si) {
            const auto* input_ptr = device_cast(input.data());
            auto* output_ptr        = device_cast(output.data());
            gs_launch(stream, 1, 1)([=](auto) __device__ {
                // this process have to be serial, so only use the first thread
                std::size_t out_idx = 0;
                for (std::size_t i = 0; i < elem_num; ++i)
                    if (not float_equal(input_ptr[i], 0))
                    {   
                        auto idx = si.multi(i);
                        for (std::size_t j = 0; j < idx.size(); ++j)
                            output_ptr[j * (*nonzero_num) + out_idx] = idx[j];
                        ++out_idx;
                    }
            });
        });
    });
    (void)hipDeviceSynchronize();

    int num = 0;
    (void)hipMemcpy(&num, nonzero_num, sizeof(int), hipMemcpyDeviceToHost);
    (void)hipFree(nonzero_num);

    auto out_lens = result.get_shape().lens();
    out_lens[1] = num;
    shape out_s{result.get_shape().type(), out_lens};

    return {out_s, result.data()};
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
