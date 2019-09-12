#include <rocblas-types.h>
#include <migraphx/gpu/gemm_impl.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

rocblas_datatype get_type(shape::type_t type)
{
    switch(type)
    {
    case shape::double_type: return rocblas_datatype_f64_r;
    case shape::float_type: return rocblas_datatype_f32_r;
    case shape::half_type: return rocblas_datatype_f16_r;
    case shape::int8_type: return rocblas_datatype_i8_r;
    case shape::uint8_type: return rocblas_datatype_u8_r;
    case shape::int32_type: return rocblas_datatype_i32_r;
    case shape::uint32_type: return rocblas_datatype_u32_r;
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
}

template<class T>
void gemm_impl(context& ctx, const shape& output_shape, const std::vector<argument>& args, T alpha, T beta)
{
    bool transa     = args[0].get_shape().transposed();
    bool transb     = args[1].get_shape().transposed();
    auto n_dim      = output_shape.lens().size();
    auto dim_1      = n_dim - 1;
    auto dim_0      = n_dim - 2;
    rocblas_int lda = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc = args[2].get_shape().strides()[dim_0];

    bool is_3inputs = (args.size() == 4);
    if(!is_3inputs)
    {
        beta = 0;
    }
    rocblas_datatype arg_type = get_type(args[0].get_shape().type());
    auto output_type          = arg_type;
    if(output_type == rocblas_datatype_i8_r)
    {
        output_type = rocblas_datatype_i32_r;
    }
    auto compute_type = output_type;

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto alpha_r    = as(alpha);
        auto beta_r     = as(beta);
        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = args[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return as.from(arg.data()); };
        if(args[0].get_shape().type() == shape::int8_type and (k % 4) != 0)
        {
            MIGRAPHX_THROW("ROCBLAS_GEMM: k size of int8 type input must be mutlple of 4!");
        }

        auto num_matrices = std::accumulate(out_lens.rbegin() + 2,
                                            out_lens.rend(),
                                            std::size_t{1},
                                            std::multiplies<std::size_t>());
        if(num_matrices == 1)
        {
            // the rocblas_gemm API handles inputs and output matrices as
            // column-major format. When doing a C = A * B, we actually do
            // C^T = (B^T) * (A^T). That is the reason we input args[1] as
            // A and args[0] as B in calling the rocblas_gemm.
            rocblas_gemm_ex(ctx.get_stream().get_rocblas(),
                            transb ? rocblas_operation_transpose : rocblas_operation_none,
                            transa ? rocblas_operation_transpose : rocblas_operation_none,
                            n,
                            m,
                            k,
                            &alpha_r,
                            to_pointer(args.at(1)),
                            arg_type,
                            ldb,
                            to_pointer(args.at(0)),
                            arg_type,
                            lda,
                            &beta_r,
                            to_pointer(args[2]),
                            output_type,
                            ldc,
                            is_3inputs ? to_pointer(args[3]) : to_pointer(args[2]),
                            output_type,
                            ldc,
                            compute_type,
                            rocblas_gemm_algo_standard,
                            0,
                            0,
                            nullptr,
                            nullptr);
        }
        else
        {
            rocblas_gemm_strided_batched_ex(
                ctx.get_stream().get_rocblas(),
                transb ? rocblas_operation_transpose : rocblas_operation_none,
                transa ? rocblas_operation_transpose : rocblas_operation_none,
                n,
                m,
                k,
                &alpha_r,
                to_pointer(args.at(1)),
                arg_type,
                ldb,
                k * n,
                to_pointer(args.at(0)),
                arg_type,
                lda,
                m * k,
                &beta_r,
                to_pointer(args[2]),
                output_type,
                ldc,
                m * n,
                is_3inputs ? to_pointer(args[3]) : to_pointer(args[2]),
                output_type,
                ldc,
                m * n,
                num_matrices,
                compute_type,
                rocblas_gemm_algo_standard,
                0,
                0,
                nullptr,
                nullptr);
        }
    });
}

void gemm(context& ctx, const shape& output_shape, const std::vector<argument>& args, float alpha, float beta)
{
    gemm_impl(ctx, output_shape, args, alpha, beta);
}

void gemm(context& ctx, const shape& output_shape, const std::vector<argument>& args, int32_t alpha, int32_t beta)
{
    gemm_impl(ctx, output_shape, args, alpha, beta);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
