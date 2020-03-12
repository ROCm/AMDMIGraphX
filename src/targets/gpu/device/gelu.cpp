#include <migraphx/gpu/device/gelu.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// x * 0.5 * (1.0 + erf(x / sqrt(2.0)))
template<class T>
auto gelu_fn(T x) __device__
{
    return x * 0.5 * (1 + ::erf(x * M_SQRT1_2));
}

void gelu(hipStream_t stream, const argument& result, const argument& arg)
{
    nary(stream, result, arg)([](auto x) __device__ {
        return gelu_fn(to_hip_type(x));
    });
}

void add_gelu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ {
        auto sum = to_hip_type(x + y);
        return gelu_fn(sum);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
