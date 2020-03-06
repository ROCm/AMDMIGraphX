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
              const argument& min_arg,
              const argument& max_arg)
{
    nary(stream, result, arg1, arg2, min_arg, max_arg)([](auto x, auto y, auto min, auto max) __device__ {
        return ::min<decltype(x + y)>(::max<decltype(x)>(min, x + y), max);
    });
}

void add_clip(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3,
              const argument& min_arg,
              const argument& max_arg)
{
    nary(stream, result, arg1, arg2, arg3, min_arg, max_arg)([](auto x, auto y, auto z, auto min, auto max) __device__ {
        return ::min<decltype(x + y + z)>(::max<decltype(x)>(min, x + y + z), max);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
