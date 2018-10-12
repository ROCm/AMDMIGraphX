
#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace gpu {
namespace device {

void add(const argument& result, const argument& arg1, const argument& arg2);

void add(const argument& result, const argument& arg1, const argument& arg2, const argument& arg3);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
