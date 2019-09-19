#include <migraphx/gpu/device/relu.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void relu(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([] __device__ (auto x) { return std::max<decltype(x)>(0, x); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
