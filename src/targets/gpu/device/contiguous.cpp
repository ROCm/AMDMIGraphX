
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, const argument& result, const argument& arg)
{
    shape s{result.get_shape().type(), result.get_shape().lens()};
    visit_all(result, arg)([&](auto output_v, auto input_v) {
        hip_visit_views(output_v, input_v, s)([&](auto output, auto input, auto standard_shape) {
            mi_gs_launch(stream,
                         standard_shape)([=](auto idx) __device__ { output[idx] = input[idx]; });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
