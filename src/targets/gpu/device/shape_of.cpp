#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/shape_of.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument shape_of(hipStream_t stream, const argument& result, const argument& ins)
{
    std::size_t nelements = result.get_shape().elements();
    result.visit([&](auto output) {
        visit_tensor_size(ins.get_shape().lens().size(), [&](auto ndim) {
            auto* outptr = device_cast(output.data());
            hip_tensor_descriptor<ndim> desc_input(ins.get_shape());
            auto lens = desc_input.multi(ins.get_shape().elements() - 1);
            gs_launch(stream, nelements)([=](auto i) { outptr[i] = lens[i] + 1; });
        });
    });

    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
