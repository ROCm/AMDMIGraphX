#include <migraph/gpu/gemm.hpp>
#include <migraph/operators.hpp>
#include <migraph/manage_ptr.hpp>
#include <migraph/gpu/miopen.hpp>
#include <utility>

namespace migraph {
namespace gpu {

shape miopen_gemm::compute_shape(const std::vector<shape>& inputs) const
{
    check_shapes{inputs, *this}.has(3);
    return op.compute_shape({inputs.at(0), inputs.at(1)});
}
argument miopen_gemm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    float alpha     = 1.0f;
    float beta      = 0.0f;
    bool transa     = args[0].get_shape().transposed();
    bool transb     = args[1].get_shape().transposed();
    rocblas_int lda = args[0].get_shape().strides()[transa ? 1 : 0];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? 1 : 0];
    rocblas_int ldc = args[2].get_shape().strides()[0];
    rocblas_int m   = output_shape.lens()[0];
    rocblas_int n   = output_shape.lens()[1];
    rocblas_int k   = args[0].get_shape().lens()[1];
    rocblas_sgemm(ctx.rbhandle.get(),
                  transb ? rocblas_operation_transpose : rocblas_operation_none,
                  transa ? rocblas_operation_transpose : rocblas_operation_none,
                  n,
                  m,
                  k,
                  &alpha,
                  args[1].implicit(),
                  ldb,
                  args[0].implicit(),
                  lda,
                  &beta,
                  args[2].implicit(),
                  ldc);
    return args[2];
}

} // namespace gpu

} // namespace migraph
