#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/where.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument where(hipStream_t stream, const argument& arg0, const argument& arg1, const argument& arg2)
{
    auto s = arg1.get_shape();
    argument result{s};
    hip_visit_all(result, arg1, arg2)([&](auto output, auto x, auto y) {
        hip_visit_all(arg0)([&](auto cond) {
            gs_launch(stream,
                      s.elements())([=](auto i) __device__ { output[i] = cond[i] ? x[i] : y[i]; });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
