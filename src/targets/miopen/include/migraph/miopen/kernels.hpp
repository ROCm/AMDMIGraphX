#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
namespace migraph {
namespace miopen {

void hip_contiguous(migraph::shape output_shape, migraph::argument arg, migraph::argument result);

} // namespace miopen

} // namespace migraph

#endif
