#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP

#include <migraph/manage_ptr.hpp>
#include <migraph/operators.hpp>
#include <migraph/config.hpp>
#include <rocblas.h>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

using rocblas_handle_ptr = MIGRAPH_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);

rocblas_handle_ptr create_rocblas_handle_ptr();
rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s);

} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
