#include <migraphx/gpu/device/where.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void where(hipStream_t stream, const argument& result, const argument& arg0, const argument& arg1, const argument& arg2)
{
    nary(stream, result, arg0, arg1, arg2)([](auto condition, auto x, auto y) __device__ { return condition ? x : y; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
