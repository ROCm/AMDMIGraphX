#include "migraphx/gpu/device/visit.hpp"
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

argument
reverse(hipStream_t stream, argument result, argument arg1, const std::vector<int64_t>& axes)
{
    auto s = arg1.get_shape();
    // auto lens             = s.lens();
    std::vector<std::size_t> axis_len(axes.begin(), axes.end());
    shape sa{shape::float_type, axis_len};
    std::size_t nelements = s.elements();
    visit_all(result, arg1)([&](auto output1, auto input1) {
        hip_visit_views(output1, input1, s)([&](auto output, auto input, auto hs) {
            hip_visit_views(sa)([&](auto daxes) {
                auto lens = hs.lens;
                gs_launch(stream, nelements)([=](auto i) __device__ {
                    auto idx    = hs.multi(i);
                    auto in_idx = idx;
                    for(auto axis : daxes.lens)
                        in_idx[axis] = lens[axis] - 1 - idx[axis];
                    output[idx] = input[in_idx];
                });
            });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
