#include <migraphx/gpu/quant_gemm.hpp>
#include <migraphx/gpu/device/pack.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/generate.hpp>

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
    std::vector<shape> input_shapes(inputs);
    input_shapes.pop_back();
    check_shapes{input_shapes}.not_broadcasted();
    bool transa = inputs[0].transposed();
    bool transb = inputs[1].transposed();

    if(!transb)
    {
        if(arg_b.empty())
        {
            auto *p_this = const_cast<miopen_quant_gemm*>(this);
            p_this->arg_b = allocate_gpu(inputs[1]);
        }
    }

    if(transa)
    {
        if(arg_a.empty())
        {
            auto *p_this = const_cast<miopen_quant_gemm*>(this);
            p_this->arg_a = allocate_gpu(inputs[0]);
        }
    }

    return op.compute_shape(input_shapes);
}

argument miopen_quant_gemm::compute(context& ctx,
                                    const shape& output_shape,
                                    const std::vector<argument>& args) const
{
    bool transa     = args[0].get_shape().transposed();
    bool transb     = args[1].get_shape().transposed();
    auto n_dim      = output_shape.lens().size();
    auto dim_1      = n_dim - 1;
    auto dim_0      = n_dim - 2;
    rocblas_int lda = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc = args[2].get_shape().strides()[dim_0];

    if(!transb)
    {
        device::pack_a(ctx.get_stream().get(), arg_b, args[1]);
    }

    // need to pack A in this scenario, use the algorithm to pack B in the
    // comment of the API
    if(transa)
    {
        device::pack_b(ctx.get_stream().get(), arg_a, args[0]);
    }

    bool is_3inputs = (args.size() == 4);
    int32_t beta    = 0;
    if(is_3inputs)
    {
        beta = op.beta;
    }

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto alpha_r    = to_rocblas_type(as(op.alpha));
        auto beta_r     = to_rocblas_type(as(beta));
        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = args[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return to_rocblas_type(as.from(arg.data())); };
        assert(k % 4 == 0);

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
                                    (!transb) ? to_pointer(arg_b) : to_pointer(args.at(1)),
                                    rocblas_datatype_i8_r,
                                    ldb,
                                    transa ? to_pointer(arg_a) : to_pointer(args.at(0)),
                                    rocblas_datatype_i8_r,
                                    lda,
                                    &beta_r,
                                    to_pointer(args[2]),
                                    rocblas_datatype_i32_r,
                                    ldc,
                                    is_3inputs ? to_pointer(args.at(3)) : to_pointer(args[2]),
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
                (!transb) ? to_pointer(arg_b) : to_pointer(args.at(1)),
                rocblas_datatype_i8_r,
                ldb,
                k * n,
                transa ? to_pointer(arg_a) : to_pointer(args.at(0)),
                rocblas_datatype_i8_r,
                lda,
                m * k,
                &beta_r,
                to_pointer(args[2]),
                rocblas_datatype_i32_r,
                ldc,
                m * n,
                is_3inputs ? to_pointer(args.at(3)) : to_pointer(args[2]),
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

    return is_3inputs ? args.at(3) : args[2];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
