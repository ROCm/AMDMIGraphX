#include <migraphx/gpu/device/add_clip.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void add_clip(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const float max,
              const float min)
{
    nary(stream, result, arg1, arg2)([max, min](auto x, auto y) __device__ {
        return ::min<decltype(x + y)>(::max<decltype(x)>(min, x + y), max);
    });
}

void add_clip(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3,
              const float max,
              const float min)
{
    nary(stream, result, arg1, arg2, arg3)([max, min](auto x, auto y, auto z) __device__ {
        return ::min<decltype(x + y + z)>(::max<decltype(x)>(min, x + y + z), max);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
