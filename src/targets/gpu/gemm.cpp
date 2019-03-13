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

void miopen_gemm::fill_result(context& ctx, const shape& output_shape, 
    const argument& result, const argument& c) const
{
    auto out_lens = output_shape.lens();
    auto c_lens = c.get_shape().lens();
    if (output_shape == c.get_shape())
    {
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg) {
                return to_rocblas_type(as.from(arg.data()));
            };
            hipMemcpy(to_pointer(args[3]),
                    to_pointer(args[2]),
                    output_shape.bytes(),
                    hipMemcpyDeviceToDevice);
        });
    }
    else if (c.single())
    {
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg, std::size_t offset) {
                return to_rocblas_type(as.from(arg.data() + offset));
            };

            for(std::size_t i = 0; i < output_shape.elements(); ++i)
            {
                hipMemcpy(to_pointer(args[3], i),
                        to_pointer(args[2]),
                        args[2].get_shape().bytes(),
                        hipMemcpyDeviceToDevice);            
            }
        });
    }
    else if (c_lens.size() == 1 ||
            (c_lens.size() == 2 && c_lens[1] == out_lens[1]))
    {
        auto m = out_lens[0];
        auto n = out_lens[1];
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg, std::size_t offset) {
                return to_rocblas_type(as.from(arg.data() + offset));
            };

            for(std::size_t i = 0; i < m; ++i)
            {
                hipMemcpy(to_pointer(args[3], i * n),
                        to_pointer(args[2]),
                        args[2].get_shape().bytes(),
                        hipMemcpyDeviceToDevice);
            }
        });
    }
    // case of c_lens.size() == 2 && c_len[0] == out_lens[0]
    else
    {
        output_shape.visit_type([&](auto as) {
            auto to_pointer = [&](auto&& arg, std::size_t offset) {
                return to_rocblas_type(as.from(arg.data() + offset));
            };

            for(std::size_t i = 0; i < output_shape.elements(); ++i)
            {
                hipMemcpy(to_pointer(args[3], i),
                        to_pointer(args[2], i / n),
                        args[2].get_shape().type_size(),
                        hipMemcpyDeviceToDevice);
            }
        });
    }
}

argument miopen_gemm::compute(context& ctx,
                              const shape& output_shape,
                              const std::vector<argument>& args) const
{
    bool is_3inputs = (args.size() == 4);
    if (is_3inputs)
    {
        fill_result(ctx, output_shape, args[3], args[2]);
        
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r     = to_rocblas_type(as(op.beta));
            bool transa        = args[0].get_shape().transposed();
            bool transb        = args[1].get_shape().transposed();
            rocblas_int lda    = args[0].get_shape().strides()[transa ? 1 : 0];
            rocblas_int ldb    = args[1].get_shape().strides()[transb ? 1 : 0];
            rocblas_int ldc    = args[2].get_shape().strides()[0];
            auto out_lens      = output_shape.lens();
            rocblas_int m      = out_lens[0];
            rocblas_int n      = out_lens[1];
            rocblas_int k      = args[0].get_shape().lens()[1];
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

            generic_rocblas_scal(as,
                                 ctx.get_stream().get_rocblas(),
                                 1,
                                 &alpha_r,
                                 to_pointer(args[2]));
                                 1);
        });
    }
    // matrix * vector
    else if (args[1].get_shape().lens().size() == 1)
    {
        auto a_lens = args[0].get_shape().lens();
        std::size_t dim_0 = a_lens.size() - 2;
        std::size_t dim_1 = a_lens.size() - 1;
        bool trans        = args[0].get_shape().transposed();
        rocblas_int m      = a_lens[trans ? dim_1 : dim_0];
        rocblas_int n      = a_lens[trans ? dim_0 : dim_1];
        float beta = 0.0f;
        rocblas_int lda    = args[0].get_shape().strides()[trans ? dim_1 : dim_0];

        assert(a_lens.back() == args[1].get_shape().elements());
        std::size_t batch_num = std::accumulate(a_lens.rbegin() + 2, a_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r =   = to_rocblas_type(as(beta));
            auto to_pointer = [&](auto&& arg, std::size_t offset = 0) { return to_rocblas_type(as.from(arg.data() + offset)); };
            for (std::size_t batch_no = 0; batch_no < batch_num; ++batch_no)
            {
                generic_rocblas_gemv(as,
                                     ctx.get_stream().get_rocblas(),
                                     trans ? rocblas_operation_transpose : rocblas_operation_none,
                                     m,
                                     n,
                                     &alpha_r,
                                     to_pointer(args[0], batch_no * m * n),
                                     lda,
                                     to_pointer(args[1]),
                                     1,
                                     &beta_r,
                                     to_pointer(args[2], batch_no * n)
                                     1);
            }
        });
    }
    // vector * matrix
    else if (args[0].get_shape().lens().size() == 1)
    {
        auto b_lens = args[1].get_shape().lens();
        std::size_t dim_0 = b_lens.size() - 2;
        std::size_t dim_1 = b_lens.size() - 1;
        bool trans        = !args[1].get_shape().transposed();
        rocblas_int m      = b_lens[trans ? dim_1 : dim_0];
        rocblas_int n      = b_lens[trans ? dim_0 : dim_1];
        float beta = 0.0f;
        rocblas_int lda    = args[1].get_shape().strides()[trans ? dim_1 : dim_0];

        assert(b_lens.back() == args[0].get_shape().elements());
        std::size_t batch_num = std::accumulate(b_lens.rbegin() + 2, b_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        output_shape.visit_type([&](auto as) {
            auto alpha_r    = to_rocblas_type(as(op.alpha));
            auto beta_r =   = to_rocblas_type(as(beta));
            auto to_pointer = [&](auto&& arg, std::size_t offset = 0) { return to_rocblas_type(as.from(arg.data() + offset)); };
            for (std::size_t batch_no = 0; batch_no < batch_num; ++batch_no)
            {
                generic_rocblas_gemv(as,
                                     ctx.get_stream().get_rocblas(),
                                     trans ? rocblas_operation_transpose : rocblas_operation_none,
                                     n,
                                     m,
                                     &alpha_r,
                                     to_pointer(args[0]),
                                     lda,
                                     to_pointer(args[1], batch_no * m * n),
                                     1,
                                     &beta_r,
                                     to_pointer(args[2], batch_no * m)
                                     1);
            }
        });
    }
    // (batch) matrix multiplication
    else
    {
        bool transa        = args[0].get_shape().transposed();
        bool transb        = args[1].get_shape().transposed();
        auto a_lens = args[0].get_shape().lens();
        auto b_lens = args[1].get_shape().lens();
        auto out_lens = output_shape.lens();

        rocblas_int lda    = args[0].get_shape().strides()[transa ? a_lens.size() - 1 : a_lens.size() - 2];
        rocblas_int ldb    = args[1].get_shape().strides()[transb ? b_lens.size() - 1 : b_lens.size() - 2];
        rocblas_int ldc    = args[2].get_shape().strides()[out_lens.size() - 2];
        rocblas_int m      = out_lens[out_lens.size() - 2];
        rocblas_int n      = out_lens[out_lens.size() - 1];
        rocblas_int k      = args[0].get_shape().lens()[a_lens.size() - 1];
        auto input_dims = std::min(a_lens.size(), b_lens.size());
        std::size_t axis{0};
        for (axis = 2; axis < input_dims; ++axis)
        {
            if (a_lens[a_lens.size() - axis] != b_lens[b_lens.size() - axis])
            {
                break;
            }
        }

        // The number of matrices that can be computed in one call
        // batch_num > 1, we need to call the batch_gemm function, 
        // otherwise, call the gemm function directly
        std::size_t num_matrices = std::accumulate(a_lens.rbegin() + 2, 
                (axis == a_lens.size() ? a_lens.rend() : a_lens.rbegin() + axis), 
                std::size_t{1}, std::multiplies<std::size_t>());
        std::size_t a_len_diff = out_lens.size() - a_lens.size();
        std::size_t b_len_diff = out_lens.size() - b_lens.size();
        std::vector<std::size_t> a_batch_lens(a_lens.begin(), a_lens.begin() + a_lens.size() - axis);
        std::vector<std::size_t> b_batch_lens(b_lens.begin(), b_lens.begin() + b_lens.size() - axis);
        std::vector<std::size_t> out_batch_lens(out_lens.begin(), out_lens.begin() + out_lens.size() - axis);

        shape::type_t t = output_shape.type();
        shape a_batch_shape{t, a_batch_lens};
        shape b_batch_shape{t, b_batch_lens};
        shape out_diff_shape{t, out_batch_lens};

        shape_for_each(out_diff_shape, [&](auto out_idx) {
            std::size_t out_ind = out_batch_shape.index(out_idx.begin(), out_idx.end());
            std::vector<std::size_t> a_idx(a_lens.size() - axis);
            std::vector<std::size_t> b_idx(b_lens.size() - axis);
            std::transform(out_idx.begin() + a_len_diff, out_idx.end(), a_batch_lens.begin(), a_idx.begin(), [&](auto i, auto j) {
                return (j == 1) ? 0 : i;
            });
            std::transform(out_idx.begin() + b_len_diff, out_idx.end(), b_batch_lens.begin(), b_idx.begin(), [&](auto i, auto j) {
                return (j == 1) ? 0 : i;
            });

            std::size_t a_ind = a_batch_shape.index(a_idx.begin(), b_idx.end());
            std::size_t b_ind = b_batch_shape.index(b_idx.begin(), b_idx.end());

            output_shape.visit_type([&](auto as) {
                auto alpha_r    = to_rocblas_type(as(op.alpha));
                auto beta_r =   = to_rocblas_type(as(beta));
                auto to_pointer = [&](auto&& arg, std::size_t offset = 0) { return to_rocblas_type(as.from(arg.data() + offset)); };
                generic_rocblas_batched_gemm(as,
                                            ctx.get_stream().get_rocblas(),
                                            transb ? rocblas_operation_transpose : rocblas_operation_none,
                                            transa ? rocblas_operation_transpose : rocblas_operation_none,
                                            n,
                                            m,
                                            k,
                                            &alpha_r,
                                            to_pointer(args[1], k * n * num_matrices * b_ind),
                                            ldb,
                                            k * n,
                                            to_pointer(args[0], m * k * num_matrices * a_ind),
                                            lda,
                                            m * k,
                                            &beta_r,
                                            to_pointer(args[2], m * n * num_matrices * out_ind),
                                            ldc,
                                            m * n,
                                            num_matrices);
            });
        });
    }

    return args[2];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
