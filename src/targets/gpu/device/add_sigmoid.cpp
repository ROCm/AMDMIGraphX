#include <migraphx/gpu/device/add_sigmoid.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void add_sigmoid(hipStream_t stream,
                 const argument& result,
                 const argument& arg1,
                 const argument& arg2)
{
    nary(stream, result, arg1, arg2)(
        [](auto x, auto y) __device__ { return 1.f / (1.f + ::exp(to_hip_type(-(x + y)))); });
}

void add_sigmoid(hipStream_t stream,
                 const argument& result,
                 const argument& arg1,
                 const argument& arg2,
                 const argument& arg3)
{
    nary(stream, result, arg1, arg2, arg3)([](auto x, auto y, auto z) __device__ {
        return 1.f / (1.f + ::exp(to_hip_type(-(x + y + z))));
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
