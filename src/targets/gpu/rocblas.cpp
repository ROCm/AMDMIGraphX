#include <migraphx/gpu/rocblas.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

rocblas_handle_ptr create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return rocblas_handle_ptr{handle};
}

rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s)
{
    rocblas_handle_ptr rb = create_rocblas_handle_ptr();
    rocblas_set_stream(rb.get(), s);
    return rb;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
