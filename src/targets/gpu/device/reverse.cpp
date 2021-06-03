#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/reverse.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument reverse(hipStream_t stream, argument result, argument arg1, int64_t axis)
{
    auto s                = arg1.get_shape();
    auto axis_dim_size    = s.lens()[axis];
    std::size_t nelements = s.elements();
    visit_all(result, arg1)([&](auto output1, auto input1) {
        hip_visit_views(output1, input1, s)([&](auto output, auto input, auto hs) {
            gs_launch(stream, nelements, 256)([=](auto i) __device__ {
                auto idx     = hs.multi(i);
                auto in_idx  = idx;
                in_idx[axis] = axis_dim_size - 1 - idx[axis];
                output[idx]  = input[in_idx];
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
