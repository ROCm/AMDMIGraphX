
#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add_relu(argument result, argument arg1, argument arg2);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
