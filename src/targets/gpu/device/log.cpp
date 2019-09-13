#include <migraphx/gpu/device/log.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void log(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([] __device__ (auto x) { return ::log(to_hip_type(x)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
