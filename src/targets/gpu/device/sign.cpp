#include <migraphx/gpu/device/sign.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void sign(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) { return (x > 0 ? 1 : ((x < 0) ? -1 : 0)); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
