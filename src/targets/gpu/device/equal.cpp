#include <migraphx/gpu/device/equal.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/type_traits.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T>
__device__ bool equal(T x, T y)
{
    auto eps  = std::numeric_limits<T>::epsilon();
    auto diff = x - y;
    return (diff <= eps) and (diff >= -eps);
}

void equal(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) __device__ { return equal(x, y); });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
