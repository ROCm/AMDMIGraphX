
#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(const argument& result, const argument& arg1, const argument& arg2);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
