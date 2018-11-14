#include <migraphx/gpu/gemm.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/miopen.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

template <class... Ts>
void generic_rocblas_gemm(shape::as<float>, Ts&&... xs)
{
    rocblas_sgemm(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_gemm(shape::as<double>, Ts&&... xs)
{
    rocblas_dgemm(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_gemm(shape::as<half>, Ts&&... xs)
{
    rocblas_hgemm(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_gemm(shape::as<T>, Ts&&...)
{
    MIGRAPH_THROW("Type unsupported by rocblas");
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
    output_shape.visit_type([&](auto as) {
        auto alpha_r    = to_rocblas_type(as(alpha));
        auto beta_r     = to_rocblas_type(as(beta));
        auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
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
                             to_pointer(args[2]),
                             ldc);

    });
    return args[2];
}

} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
