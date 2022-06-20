#include <migraphx/gpu/device/where.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/launch.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class Shape>
constexpr auto get_rank(const Shape&)
{
    return decltype(typename Shape::hip_index{}.size()){};
}

void where(hipStream_t stream,
           const argument& result,
           const argument& arg0,
           const argument& arg1,
           const argument& arg2)
{
    hip_visit_all(result, arg1, arg2)([&](auto output, auto x, auto y) {
        hip_visit_all(arg0)([&](auto cond) {
            if constexpr(get_rank(cond.get_shape()) == get_rank(output.get_shape()))
            {
                gs_launch(stream, arg1.get_shape().elements())([=](auto idx) __device__ {
                    auto i    = output.get_shape().multi(idx);
                    output[i] = cond[i] ? x[i] : y[i];
                });
            }
        });
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
