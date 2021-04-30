#include <migraphx/gpu/device/clip.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/math.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void clip(hipStream_t stream,
          const argument& result,
          const argument& arg1,
          const argument& min_val,
          const argument& max_val)
{

    nary(stream, result, arg1, min_val, max_val)([](auto x, auto min_v, auto max_v) __device__ {
        return min<decltype(x)>(max<decltype(x)>(min_v, x), max_v);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
