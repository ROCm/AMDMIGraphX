#include <migraphx/gpu/device/cosh.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void cosh(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ { return ::cosh(to_hip_type(x)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
