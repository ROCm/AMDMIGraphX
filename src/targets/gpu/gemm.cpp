#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/add.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class... Ts>
rocblas_status generic_rocblas_scal(shape::as<float>, Ts&&... xs)
{
    return rocblas_sscal(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_scal(shape::as<double>, Ts&&... xs)
{
    return rocblas_dscal(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_scal(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_SCAL: type unsupported by rocblas");
}

template <class... Ts>
rocblas_status generic_rocblas_axpy(shape::as<half>, Ts&&... xs)
{
    return rocblas_haxpy(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_axpy(shape::as<float>, Ts&&... xs)
{
    return rocblas_saxpy(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_axpy(shape::as<double>, Ts&&... xs)
{
    return rocblas_daxpy(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_axpy(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_AXPY: type unsupported by rocblas");
}

template <class... Ts>
rocblas_status generic_rocblas_dot(shape::as<float>, Ts&&... xs)
{
    return rocblas_sdot(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_dot(shape::as<double>, Ts&&... xs)
{
    return rocblas_ddot(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_dot(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_DOT: type unsupported by rocblas");
}

template <class... Ts>
rocblas_status generic_rocblas_gemv(shape::as<float>, Ts&&... xs)
{
    return rocblas_sgemv(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_gemv(shape::as<double>, Ts&&... xs)
{
    return rocblas_dgemv(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_gemv(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_GEMMV: type unsupported by rocblas");
}

template <class... Ts>
rocblas_status generic_rocblas_batched_gemm(shape::as<float>, Ts&&... xs)
{
    return rocblas_sgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_batched_gemm(shape::as<double>, Ts&&... xs)
{
    return rocblas_dgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_batched_gemm(shape::as<half>, Ts&&... xs)
{
    return rocblas_hgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_batched_gemm(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_BATCHED_GEMM: type unsupported by rocblas");
}

template <class... Ts>
rocblas_status generic_rocblas_gemm(shape::as<float>, Ts&&... xs)
{
    return rocblas_sgemm(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_gemm(shape::as<double>, Ts&&... xs)
{
    return rocblas_dgemm(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_gemm(shape::as<half>, Ts&&... xs)
{
    return rocblas_hgemm(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
rocblas_status generic_rocblas_gemm(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_GEMM: type unsupported by rocblas");
}

template <class T>
struct compute_rocblas_type
{
    using type = T;
};

template <class T>
struct compute_rocblas_type<const T>
{
    using type = const typename compute_rocblas_type<T>::type;
};

template <>
struct compute_rocblas_type<half>
{
    using type = rocblas_half;
};

template <class T>
using rb_type = typename compute_rocblas_type<T>::type;

template <class T>
rb_type<T> to_rocblas_type(T x)
{
    return reinterpret_cast<const rb_type<T>&>(x);
}

template <class T>
rb_type<T>* to_rocblas_type(T* x)
{
    return reinterpret_cast<rb_type<T>*>(x);
}

rocblas_half to_rocblas_type(half x) { return reinterpret_cast<const rocblas_half&>(x); }

shape miopen_gemm::compute_shape(const std::vector<shape>& inputs) const
{
    std::vector<shape> input_shapes(inputs.begin(), inputs.begin() + inputs.size() - 1);
    check_shapes{input_shapes}.standard();
    return op.compute_shape(input_shapes);
}

argument miopen_gemm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    bool is_3inputs = (args.size() == 4);
    float beta      = 0.0f;
    if(is_3inputs)
    {
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
            hipMemcpyAsync(to_pointer(args[3]),
                           to_pointer(args[2]),
                           output_shape.bytes(),
                           hipMemcpyDeviceToDevice,
                           ctx.get_stream().get());
        });
        beta = op.beta;
    }

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto n_dim        = output_shape.lens().size();
        auto dim_1        = n_dim - 1;
        auto dim_0        = n_dim - 2;
        auto alpha_r      = to_rocblas_type(as(op.alpha));
        auto beta_r       = to_rocblas_type(as(beta));
        bool transa       = args[0].get_shape().transposed();
        bool transb       = args[1].get_shape().transposed();
        rocblas_int lda   = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
        rocblas_int ldb   = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
        rocblas_int ldc   = args[2].get_shape().strides()[dim_0];
        auto out_lens     = output_shape.lens();
        rocblas_int m     = out_lens[dim_0];
        rocblas_int n     = out_lens[dim_1];
        rocblas_int k     = args[0].get_shape().lens()[dim_1];
        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
        if(num_matrices == 1)
        {
            generic_rocblas_gemm(as,
                                 ctx.get_stream().get_rocblas(),
                                 transb ? rocblas_operation_transpose : rocblas_operation_none,
                                 transa ? rocblas_operation_transpose : rocblas_operation_none,
                                 n,
                                 m,
                                 k,
                                 &alpha_r,
                                 to_pointer(args[1]),
                                 ldb,
                                 to_pointer(args[0]),
                                 lda,
                                 &beta_r,
                                 (is_3inputs ? to_pointer(args[3]) : to_pointer(args[2])),
                                 ldc);
        }
        else
        {
            generic_rocblas_batched_gemm(
                as,
                ctx.get_stream().get_rocblas(),
                transb ? rocblas_operation_transpose : rocblas_operation_none,
                transa ? rocblas_operation_transpose : rocblas_operation_none,
                n,
                m,
                k,
                &alpha_r,
                to_pointer(args[1]),
                ldb,
                k * n,
                to_pointer(args[0]),
                lda,
                m * k,
                &beta_r,
                (is_3inputs ? to_pointer(args[3]) : to_pointer(args[2])),
                ldc,
                m * n,
                num_matrices);
        }
    });

    return (is_3inputs ? args[3] : args[2]);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
