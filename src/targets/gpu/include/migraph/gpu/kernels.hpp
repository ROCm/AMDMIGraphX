#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
namespace migraph {
namespace gpu {

void hip_contiguous(migraph::shape output_shape, migraph::argument arg, migraph::argument result);

} // namespace gpu

} // namespace migraph

#endif
