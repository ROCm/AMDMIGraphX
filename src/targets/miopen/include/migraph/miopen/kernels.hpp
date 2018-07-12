#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
namespace migraph {
namespace miopen {

migraph::argument hip_contiguous(migraph::argument arg, migraph::shape output_shape);

} // namespace miopen

} // namespace migraph

#endif
