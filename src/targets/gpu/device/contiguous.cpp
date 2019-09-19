
#include <migraphx/gpu/device/contiguous.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, argument result, argument arg)
{
    nary_nonstandard(stream, std::move(result), std::move(arg))([] __device__ (auto x) { return x; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
