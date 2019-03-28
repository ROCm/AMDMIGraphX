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
    std::vector<shape> input_shapes(inputs.begin(), inputs.begin() + inputs.size() - 1);
    if(input_shapes.size() == 3)
    {
        check_shapes{{input_shapes.back()}}.not_broadcasted();
    }
    return op.compute_shape(input_shapes);
}

argument miopen_gemm::batch_matmul(context& ctx,
                                   const shape& output_shape,
                                   const std::vector<argument>& args) const
{
    bool transa = args[0].get_shape().transposed();
    bool transb = args[1].get_shape().transposed();

    auto a_lens   = args[0].get_shape().lens();
    auto b_lens   = args[1].get_shape().lens();
    auto out_lens = output_shape.lens();

    auto an_dim   = a_lens.size();
    auto bn_dim   = b_lens.size();
    auto outn_dim = out_lens.size();

    rocblas_int lda = args[0].get_shape().strides()[transa ? an_dim - 1 : an_dim - 2];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? bn_dim - 1 : bn_dim - 2];
    rocblas_int ldc = args[2].get_shape().strides()[outn_dim - 2];
    rocblas_int m   = out_lens[outn_dim - 2];
    rocblas_int n   = out_lens[outn_dim - 1];
    rocblas_int k   = a_lens[an_dim - 1];
    float beta      = 0.0f;

    std::vector<std::size_t> a_batch_lens(a_lens.begin(), a_lens.begin() + an_dim - 2);
    std::vector<std::size_t> b_batch_lens(b_lens.begin(), b_lens.begin() + bn_dim - 2);
    if(a_batch_lens == b_batch_lens || a_batch_lens.empty() || b_batch_lens.empty())
    {
        std::size_t numa_matrices = std::accumulate(a_batch_lens.begin(),
                                                    a_batch_lens.end(),
                                                    std::size_t{1},
                                                    std::multiplies<std::size_t>());
        std::size_t numb_matrices = std::accumulate(b_batch_lens.begin(),
                                                    b_batch_lens.end(),
                                                    std::size_t{1},
                                                    std::multiplies<std::size_t>());
        std::size_t num_matrices  = std::max(numa_matrices, numb_matrices);
        rocblas_int stride_a      = (numa_matrices == 1) ? 0 : m * k;
        rocblas_int stride_b      = (numb_matrices == 1) ? 0 : k * n;
        rocblas_int stride_c      = m * n;
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(beta));
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
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
                stride_b,
                to_pointer(args[0]),
                lda,
                stride_a,
                &beta_r,
                to_pointer(args[2]),
                ldc,
                stride_c,
                num_matrices);
        });
    }
    else
    {
        std::vector<std::size_t> out_batch_lens(out_lens.begin(), out_lens.begin() + outn_dim - 2);
        shape::type_t t = output_shape.type();
        shape a_batch_shape{t, a_batch_lens};
        shape b_batch_shape{t, b_batch_lens};
        shape out_batch_shape{t, out_batch_lens};
        std::size_t a_len_diff = outn_dim - an_dim;
        std::size_t b_len_diff = outn_dim - bn_dim;

        shape_for_each(out_batch_shape, [&](auto out_idx) {
            std::size_t out_ind = out_batch_shape.index(out_idx.begin(), out_idx.end());
            auto type_size      = output_shape.type_size();
            std::vector<std::size_t> a_idx(a_batch_lens.size());
            std::vector<std::size_t> b_idx(b_batch_lens.size());
            std::transform(out_idx.begin() + a_len_diff,
                           out_idx.end(),
                           a_batch_lens.begin(),
                           a_idx.begin(),
                           [&](auto i, auto j) { return (j == 1) ? 0 : i; });
            std::transform(out_idx.begin() + b_len_diff,
                           out_idx.end(),
                           b_batch_lens.begin(),
                           b_idx.begin(),
                           [&](auto i, auto j) { return (j == 1) ? 0 : i; });

            std::size_t a_ind = a_batch_shape.index(a_idx.begin(), a_idx.end());
            std::size_t b_ind = b_batch_shape.index(b_idx.begin(), b_idx.end());

            output_shape.visit_type([&](auto as) {
                auto alpha_r    = to_rocblas_type(as(op.alpha));
                auto beta_r     = to_rocblas_type(as(beta));
                auto to_pointer = [&](auto&& arg, std::size_t offset = 0) {
                    return to_rocblas_type(as.from(arg.data() + offset));
                };
                generic_rocblas_gemm(as,
                                     ctx.get_stream().get_rocblas(),
                                     transb ? rocblas_operation_transpose : rocblas_operation_none,
                                     transa ? rocblas_operation_transpose : rocblas_operation_none,
                                     n,
                                     m,
                                     k,
                                     &alpha_r,
                                     to_pointer(args[1], k * n * b_ind * type_size),
                                     ldb,
                                     to_pointer(args[0], m * k * a_ind * type_size),
                                     lda,
                                     &beta_r,
                                     to_pointer(args[2], m * n * out_ind * type_size),
                                     ldc);
            });
        });
    }

    return args[2];
}

argument miopen_gemm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    bool is_3inputs = (args.size() == 4);
    if(is_3inputs)
    {
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
            hipMemcpy(to_pointer(args[3]),
                      to_pointer(args[2]),
                      output_shape.bytes(),
                      hipMemcpyDeviceToDevice);
        });

        // fill_result(output_shape, args[3], args[2]);
        output_shape.visit_type([&](auto as) {
            auto n_dim        = output_shape.lens().size();
            auto dim_1        = n_dim - 1;
            auto dim_0        = n_dim - 2;
            auto alpha_r      = to_rocblas_type(as(op.alpha));
            auto beta_r       = to_rocblas_type(as(op.beta));
            bool transa       = args[0].get_shape().transposed();
            bool transb       = args[1].get_shape().transposed();
            rocblas_int lda   = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
            rocblas_int ldb   = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
            rocblas_int ldc   = args[3].get_shape().strides()[dim_0];
            auto out_lens     = output_shape.lens();
            rocblas_int m     = out_lens[dim_0];
            rocblas_int n     = out_lens[dim_1];
            rocblas_int k     = args[0].get_shape().lens()[dim_1];
            auto num_matrices = std::accumulate(out_lens.rbegin() + 2,
                                                out_lens.rend(),
                                                std::size_t{1},
                                                std::multiplies<std::size_t>());
            auto to_pointer   = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
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
                to_pointer(args[3]),
                ldc,
                m * n,
                num_matrices);

        });

        return args[3];
    }

    // 2 input arguments cases
    // vector inner product
    if(output_shape.elements() == 1)
    {
        assert(args[0].get_shape().elements() == args[1].get_shape().elements());
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
            generic_rocblas_dot(as,
                                ctx.get_stream().get_rocblas(),
                                args[1].get_shape().elements(),
                                to_pointer(args[0]),
                                1,
                                to_pointer(args[1]),
                                1,
                                to_pointer(args[2]));

            generic_rocblas_scal(
                as, ctx.get_stream().get_rocblas(), 1, &alpha_r, to_pointer(args[2]), 1);
        });
    }
    // matrix * vector
    else if(args[1].get_shape().lens().size() == 1)
    {
        auto a_lens       = args[0].get_shape().lens();
        auto b_lens       = args[1].get_shape().lens();
        std::size_t dim_0 = a_lens.size() - 2;
        std::size_t dim_1 = a_lens.size() - 1;
        bool transa       = args[0].get_shape().transposed();
        bool transb       = false;
        rocblas_int lda   = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
        rocblas_int ldb   = 1;
        rocblas_int ldc   = 1;
        rocblas_int m     = a_lens[dim_0];
        rocblas_int n     = 1;
        rocblas_int k     = a_lens[dim_1];
        float beta        = 0.0f;
        assert(a_lens.back() == args[1].get_shape().elements());

        std::size_t batch_num = std::accumulate(
            a_lens.rbegin() + 2, a_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(beta));
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };

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
                0,
                to_pointer(args[0]),
                lda,
                m * k,
                &beta_r,
                to_pointer(args[2]),
                ldc,
                m * n,
                batch_num);
        });
    }
    // vector * matrix
    else if(args[0].get_shape().lens().size() == 1)
    {
        auto a_lens       = args[0].get_shape().lens();
        auto b_lens       = args[1].get_shape().lens();
        std::size_t dim_0 = b_lens.size() - 2;
        std::size_t dim_1 = b_lens.size() - 1;
        bool transb       = args[1].get_shape().transposed();
        bool transa       = false;
        rocblas_int lda   = a_lens[0];
        rocblas_int ldb   = args[1].get_shape().strides()[(transb ? dim_1 : dim_0)];
        rocblas_int ldc   = b_lens[dim_1];
        rocblas_int m     = 1;
        rocblas_int n     = args[1].get_shape().lens()[dim_1];
        rocblas_int k     = a_lens[0];
        float beta        = 0.0f;
        assert(b_lens[dim_0] == args[0].get_shape().elements());

        std::size_t batch_num = std::accumulate(
            b_lens.rbegin() + 2, b_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());

        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(beta));
            auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };

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
                0,
                &beta_r,
                to_pointer(args[2]),
                ldc,
                m * n,
                batch_num);
        });
    }
    // (batch) matrix multiplication
    else
    {
        batch_matmul(ctx, output_shape, args);
    }

    return args[2];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
