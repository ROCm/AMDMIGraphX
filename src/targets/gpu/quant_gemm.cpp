#include <migraphx/gpu/quant_gemm.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

template <class... Ts>
rocblas_status generic_rocblas_gemm_ex(Ts&&... xs)
{
    return rocblas_gemm_ex(std::forward<Ts>(xs)...);
}

template <class... Ts>
rocblas_status generic_rocblas_batched_gemm_ex(Ts&&... xs)
{
    return rocblas_gemm_strided_batched_ex(std::forward<Ts>(xs)...);
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

shape miopen_quant_gemm::compute_shape(const std::vector<shape>& inputs) const
{
    std::vector<shape> input_shapes(inputs.begin(), inputs.begin() + inputs.size() - 1);
    check_shapes{input_shapes}.not_broadcasted();
    return op.compute_shape(input_shapes);
}

argument miopen_quant_gemm::compute(context& ctx,
                                    const shape& output_shape,
                                    const std::vector<argument>& args) const
{
    bool is_3inputs = (args.size() == 4);
    int8_t beta     = 0;
    if(is_3inputs)
    {
        beta = op.beta;
    }

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto n_dim      = output_shape.lens().size();
        auto dim_1      = n_dim - 1;
        auto dim_0      = n_dim - 2;
        auto alpha_r    = to_rocblas_type(as(op.alpha));
        auto beta_r     = to_rocblas_type(as(beta));
        bool transa     = args[0].get_shape().transposed();
        bool transb     = args[1].get_shape().transposed();
        rocblas_int lda = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
        rocblas_int ldb = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
        rocblas_int ldc = args[2].get_shape().strides()[dim_0];
        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = args[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
        assert(k % 4 == 0);
        assert(!transa or (lda % 4 == 0));
        assert(transb or (ldb % 4 == 0));

        // need to pack B in thi scenario
        if (!transb)
        {
            int nb = 4;
            for(int i_m = 0; i_m < m; i_m++)
            {
                for(int i_k = 0; i_k < k; i_k++)
                {
                    A_packed[i_k % nb + (i_m + (i_k / nb) * lda) * nb] = A[i_m + i_k * lda];
                }
            }
        }

        // need to pack A in this scenario
        if (transa)
        {

        }

        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        if(num_matrices == 1)
        {
            // the rocblas_gemm API handles inputs and output matrices as 
            // column-major format. When doing a C = A * B, we actually do
            // C^T = (B^T) * (A^T). That is the reason we input args[1] as
            // A and args[0] as B in calling the rocblas_gemm.
            generic_rocblas_gemm_ex(ctx.get_stream().get_rocblas(),
                                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                                    n,
                                    m,
                                    k,
                                    &alpha_r,
                                    to_pointer(args[1]),
                                    rocblas_datatype_i8_r,
                                    ldb,
                                    to_pointer(args[0]),
                                    rocblas_datatype_i8_r,
                                    lda,
                                    &beta_r,
                                    to_pointer(args[2]),
                                    rocblas_datatype_i32_r,
                                    ldc,
                                    (is_3inputs ? to_pointer(args[3]) : to_pointer(args[2])),
                                    rocblas_datatype_i32_r,
                                    ldc,
                                    rocblas_datatype_i32_r,
                                    rocblas_gemm_algo_standard,
                                    0,
                                    0,
                                    nullptr,
                                    nullptr);
        }
        else
        {
            generic_rocblas_batched_gemm_ex(
                ctx.get_stream().get_rocblas(),
                transb ? rocblas_operation_transpose : rocblas_operation_none,
                transa ? rocblas_operation_transpose : rocblas_operation_none,
                n,
                m,
                k,
                &alpha_r,
                to_pointer(args[1]),
                rocblas_datatype_i8_r,
                ldb,
                k * n,
                to_pointer(args[0]),
                rocblas_datatype_i8_r,
                lda,
                m * k,
                &beta_r,
                to_pointer(args[2]),
                rocblas_datatype_i32_r,
                ldc,
                m * n,
                (is_3inputs ? to_pointer(args[3]) : to_pointer(args[2])),
                rocblas_datatype_i32_r,
                ldc,
                m * n,
                num_matrices,
                rocblas_datatype_i32_r,
                rocblas_gemm_algo_standard,
                0,
                0,
                nullptr,
                nullptr);
        }
    });

    return (is_3inputs ? args[3] : args[2]);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
