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
    arg_op(argmin_op{}, stream, result, arg, axis);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
