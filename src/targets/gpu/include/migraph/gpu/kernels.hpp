#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP

#include <migraph/argument.hpp>

namespace migraph {
namespace gpu {

void hip_contiguous(shape output_shape, argument arg, argument result);

} // namespace gpu

} // namespace migraph

#endif
