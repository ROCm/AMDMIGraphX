#include <migraphx/gpu/device/fill.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void fill(hipStream_t stream, const argument& result, unsigned long val)
{
    nary(stream, result)([=]() __device__ { return val; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
