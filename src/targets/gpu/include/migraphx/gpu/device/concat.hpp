#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_CONCAT_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_CONCAT_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

argument concat(hipStream_t stream,
                const shape& output_shape,
                std::vector<argument> args,
                std::vector<std::size_t> offsets);

} // namespace device
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
