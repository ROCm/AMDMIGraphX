#include <migraph/gpu/rocblas.hpp>

namespace migraph {
namespace miopen {

rocblas_handle_ptr create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return rocblas_handle_ptr{handle};
}

} // namespace miopen

} // namespace migraph
