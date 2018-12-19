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
    visit_all(result, ins)([&](auto output, auto input) {
        auto* outptr      = device_cast(output.data());
        const auto* inptr = device_cast(input.data());
        gs_launch(stream, nelements)([=](auto i) { outptr[i] = inptr[i]; });
    });
    return result;
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
