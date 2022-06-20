#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_CONCAT_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_CONCAT_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument concat(hipStream_t stream,
                const shape& output_shape,
                std::vector<argument> args,
                std::vector<std::size_t> offsets);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
