#ifndef MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_ROCBLAS_HPP

#include <migraphx/manage_ptr.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/config.hpp>
#include <rocblas.h>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

using rocblas_handle_ptr = MIGRAPH_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);

rocblas_handle_ptr create_rocblas_handle_ptr();
rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s);

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
