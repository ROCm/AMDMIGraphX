#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHER_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHER_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// use algorithm of onnx::gather (not used for now)
argument gather(hipStream_t stream,
                const migraphx::shape& output_shape,
                std::vector<migraphx::argument> args,
                std::size_t axis);

// use algorithm of torch.nn.gather
argument gather_torch(hipStream_t stream,
                      const migraphx::shape& output_shape,
                      std::vector<migraphx::argument> args,
                      std::size_t axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
