#include <migraphx/gpu/device/add_unary.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void mul_add(hipStream_t stream,
             const argument& result,
             const argument& arg1,
             const argument& arg2,
             const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)([] __device__ (auto x, auto a, auto b) { return a * x + b; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
