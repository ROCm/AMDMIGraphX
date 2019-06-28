#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/device/argmin.hpp>
#include <migraphx/gpu/device/tensor.hpp>
#include <migraphx/gpu/device/launch.hpp>
#include <migraphx/gpu/device/types.hpp>
#include <migraphx/gpu/device/arg_op.hpp>
#include <migraphx/gpu/hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void argmin(hipStream_t stream, const argument& result, const argument& arg, int axis)
{
    arg.visit([&](auto input) {
        using type     = device_type<std::remove_cv_t<typename decltype(input)::value_type>>;
        arg_op<pair_min<type, int64_t>>(pair_min<type, int64_t>{}, stream, result, arg, axis);
    });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
