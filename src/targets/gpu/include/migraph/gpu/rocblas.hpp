#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP

#include <migraph/manage_ptr.hpp>
#include <migraph/operators.hpp>
#include <rocblas.h>

namespace migraph {
namespace gpu {

using rocblas_handle_ptr = MIGRAPH_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);

rocblas_handle_ptr create_rocblas_handle_ptr();

} // namespace gpu

} // namespace migraph

#endif
