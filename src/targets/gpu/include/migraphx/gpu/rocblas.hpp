#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ROCBLAS_HPP

#include <migraphx/manage_ptr.hpp>
#include <migraphx/config.hpp>
#include <rocblas.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using rocblas_handle_ptr = MIGRAPHX_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);

rocblas_handle_ptr create_rocblas_handle_ptr();
rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
