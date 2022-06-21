#include <migraphx/gpu/device/atan.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void atan(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return ::atan(to_hip_type(x)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
