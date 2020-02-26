#include <migraphx/gpu/device/max.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void max(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    nary(stream, result, arg1, arg2)(
        [](auto x, auto y) __device__ { return ::max(to_hip_type(x), to_hip_type(y)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
