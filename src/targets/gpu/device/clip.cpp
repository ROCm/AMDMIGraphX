#include <migraphx/gpu/device/clip.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void clip(hipStream_t stream,
          const argument& result,
          const argument& arg1,
          const float max,
          const float min)
{
    nary(stream, result, arg1)([max, min](auto x) __device__ {
        return ::min<decltype(x)>(::max<decltype(x)>(min, x), max);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
