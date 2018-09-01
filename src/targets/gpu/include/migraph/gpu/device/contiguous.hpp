#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace gpu {
namespace device {

void contiguous(argument result, argument arg);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
