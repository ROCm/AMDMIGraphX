#include <migraphx/gpu/device/logical_or.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/type_traits.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void logical_or(hipStream_t stream,
                const argument& result,
                const argument& arg1,
                const argument& arg2)
{
    nary(stream, result, arg1, arg2)(
        [](auto x, auto y) __device__ { return static_cast<bool>(x) or static_cast<bool>(y); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
