#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class... Ts>
void generic_rocblas_scal(shape::as<float>, Ts&&... xs)
{
    rocblas_sscal(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_scal(shape::as<double>, Ts&&... xs)
{
    rocblas_dscal(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_scal(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_SCAL: type unsupported by rocblas");
}

template <class... Ts>
void generic_rocblas_axpy(shape::as<half>, Ts&&... xs)
{
    rocblas_haxpy(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_axpy(shape::as<float>, Ts&&... xs)
{
    rocblas_saxpy(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_axpy(shape::as<double>, Ts&&... xs)
{
    rocblas_daxpy(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_axpy(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_AXPY: type unsupported by rocblas");
}

template <class... Ts>
void generic_rocblas_dot(shape::as<float>, Ts&&... xs)
{
    rocblas_sdot(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_dot(shape::as<double>, Ts&&... xs)
{
    rocblas_ddot(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_dot(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_DOT: type unsupported by rocblas");
}

template <class... Ts>
void generic_rocblas_gemv(shape::as<float>, Ts&&... xs)
{
    rocblas_sgemv(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_gemv(shape::as<double>, Ts&&... xs)
{
    rocblas_dgemv(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_gemv(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_GEMMV: type unsupported by rocblas");
}

template <class... Ts>
void generic_rocblas_batched_gemm(shape::as<float>, Ts&&... xs)
{
    rocblas_sgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_batched_gemm(shape::as<double>, Ts&&... xs)
{
    rocblas_dgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class... Ts>
void generic_rocblas_batched_gemm(shape::as<half>, Ts&&... xs)
{
    rocblas_hgemm_strided_batched(std::forward<Ts>(xs)...);
}

template <class T, class... Ts>
void generic_rocblas_batched_gemm(shape::as<T>, Ts&&...)
{
    MIGRAPHX_THROW("GENERIC_ROCBLAS_BATCHED_GEMM: type unsupported by rocblas");
}

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
    return op.compute_shape(inputs);
}

std::size_t miopen_gemm::compute_offset(std::vector<std::size_t>& out_lens, 
    std::size_t index, std::vector<std::size_t> &data_lens) const
{
    
}

argument miopen_gemm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    bool is_3inputs = (args.size() == 4);

    if (output_shape.elements() == 1)
    {
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(op.beta));
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
            generic_rocblas_dot(as, ctx.get_stream().get_rocblas(),
                                args[1].get_shape().elements(), 
                                to_pointer(args[0]),
                                1,
                                to_pointer(args[1]),
                                1,
                                is_3inputs ? to_pointer(args[3]): to_pointer(args[2]));

            generic_rocblas_scal(as, ctx.get_stream().get_rocblas(),
                                    1, 
                                    &alpha_r, 
                                    is_3inputs ? to_pointer(args[3]): to_pointer(args[2]));
                                    1);

            if (is_3inputs)
            {

                generic_rocblas_axpy(as, ctx.get_stream().get_rocblas(),
                                     1,
                                     &beta_r,
                                     to_pointer(args[2]),
                                     1,
                                     to_pointer(args[3]),
                                     1);
            }
        });

        return is_3inputs ? args[3] : args[2];
    }

    // b is a vector, so the computation is matrix * vector
    // could not be the case of inner product of vectors since 
    // it is already processed above
    if (args[1].get_shape().lens().size() == 1)
    {
        // considering the batch input, so A could be a batch
        // of matrices
        auto a_lens = args[0].get_shape().lens();
        std::size_t n_dims = a_lens.size();
        std::size_t dim_0  = n_dims - 2;
        std::size_t dim_1  = n_dims - 1;
        bool transa        = args[0].get_shape().transposed();
        rocblas_int lda    = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
        rocblas_int m      = a_lens[dim_0];
        rocblas_int k      = a_lens[dim_1];
        auto batch_num     = std::accumulate(
            a_lens.rbegin() + 2, a_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(op.beta));
            auto to_pointer = [&](auto&& arg, std::size_t offset) { return to_rocblas_type(as.from(arg.data() + offset)); };
            for (std::size_t batch_no = 0; batch_no < batch_num; ++batch_no)
            {
                if(is_3inputs)
                    hipMemcpy(to_pointer(args[3] + batch_no * m),
                            to_pointer(args[2]),
                            output_shape.bytes(),
                            hipMemcpyDeviceToDevice);
                else
                    hipMemset(to_pointer(args[2]), 0, output_shape.bytes());
            }
        });
    }

    bool transa        = args[0].get_shape().transposed();
    bool transb        = args[1].get_shape().transposed();
    std::size_t n_dims = args[0].get_shape().lens().size();
    std::size_t dim_0  = n_dims - 2;
    std::size_t dim_1  = n_dims - 1;
    rocblas_int lda    = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb    = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc    = args[2].get_shape().strides()[dim_0];
    auto out_lens      = output_shape.lens();
    rocblas_int m      = out_lens[dim_0];
    rocblas_int n      = out_lens[dim_1];
    rocblas_int k      = args[0].get_shape().lens()[dim_1];
    auto batch_num     = std::accumulate(
        out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());

    bool is_3inputs = (args.size() == 4);
    // two input arguments
    if (!is_3inputs)
    {
    }

    output_shape.visit_type([&](auto as) {
        auto to_pointer = [&](auto&& arg, std::size_t offset = 0) { return to_rocblas_type(as.from(arg.data() + offset)); };
        if(is_3inputs)
            hipMemcpy(to_pointer(args[3]),
                      to_pointer(args[2]),
                      output_shape.bytes(),
                      hipMemcpyDeviceToDevice);
        else
            hipMemset(to_pointer(args[2]), 0, output_shape.bytes());
    });

    output_shape.visit_type([&](auto as) {
        auto to_pointer = [&](auto&& arg, std::size_t offset = 0) { return to_rocblas_type(as.from(arg.data() + offset)); };
        generic_rocblas_batched_gemm(as,
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
                                     to_pointer(args[2]),
                                     ldc,
                                     m * n,
                                     batch_num);
    });

    return (is_3inputs ? args[3] : args[2]);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
