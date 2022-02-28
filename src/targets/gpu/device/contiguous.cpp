
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <hip/hip_fp16.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void contiguous_nonstandard(hipStream_t stream, const argument& result, const argument& arg)
{
    shape s{result.get_shape().type(), result.get_shape().lens()};
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        hip_visit_views(output_v, input_v, s)([&](auto output, auto input, auto standard_shape) {
            gs_launch(stream, s.elements())([=](auto i) __device__ {
                auto idx    = standard_shape.multi(i);
                output[idx] = input[idx];
            });
            // mi_gs_launch(stream,
            //              standard_shape)([=](auto idx) __device__ { output[idx] = input[idx]; });
        });
    });
}

void contiguous_packed(hipStream_t stream, const argument& result, const argument& arg)
{
    index_int nelements = result.get_shape().elements();
    // auto type = result.get_shape().type();
    // if (type == shape::half_type)
    // {
    //     visit_all(result, arg)([&](auto output_v, auto input_v) {
    //         const auto* input = device_cast(input_v.data());
    //         auto* output      = device_cast(output_v.data());
    //         const __half2* input2 = reinterpret_cast<__half2*>(input_v.data());
    //         __half2* output2 = reinterpret_cast<__half2*>(output_v.data());
    //         gs_launch(stream, nelements / 2)([=](auto i) __device__ {
    //             output2[i] = input2[i];
    //             if (i == 0 and (nelements % 2) == 1)
    //             {
    //                 output[nelements - 1] = input[nelements - 1];
    //             }
    //         });
    //     });
    // }
    // else
    // {
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        const auto* input = device_cast(input_v.data());
        auto* output      = device_cast(output_v.data());
        gs_launch(stream, nelements)([=](auto i) __device__ { output[i] = input[i]; });
    });
    // }
}

void contiguous(hipStream_t stream, const argument& result, const argument& arg)
{
    if(result.get_shape() == arg.get_shape() and result.get_shape().packed())
        contiguous_packed(stream, result, arg);
    else
        contiguous_nonstandard(stream, result, arg);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
