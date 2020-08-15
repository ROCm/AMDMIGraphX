#include <migraphx/gpu/device/equal.hpp>
#include <migraphx/gpu/device/nary.hpp>
#include <migraphx/type_traits.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template<class T>
__device__ bool equal(T x, T y)
{
    auto eps = std::numeric_limits<T>::epsilon();
    auto diff = x - y;
    return (diff <= eps) and (diff >= -eps);
}

void equal(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    result.visit([&](auto output) {
        visit_all(arg1, arg2)([&](auto in1, auto in2) {
            const auto* in1_ptr = device_cast(in1.data());
            const auto* in2_ptr = device_cast(in2.data());
            auto* out_ptr       = device_cast(output.data());
            gs_launch(stream, result.get_shape().elements())(
                [=](auto i) __device__ { out_ptr[i] = equal(in1_ptr[i], in2_ptr[i]); });
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
