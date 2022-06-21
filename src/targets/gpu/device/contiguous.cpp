
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void contiguous_nonstandard(hipStream_t stream, const argument& result, const argument& arg)
{
    shape s{result.get_shape().type(), result.get_shape().lens()};
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        hip_visit_views(output_v, input_v, s)([&](auto output, auto input, auto standard_shape) {
            mi_gs_launch(stream,
                         standard_shape)([=](auto idx) __device__ { output[idx] = input[idx]; });
        });
    });
}

void contiguous_packed(hipStream_t stream, const argument& result, const argument& arg)
{
    index_int nelements = result.get_shape().elements();
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        const auto* input = device_cast(input_v.data());
        auto* output      = device_cast(output_v.data());
        gs_launch(stream, nelements)([=](auto i) __device__ { output[i] = input[i]; });
    });
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
