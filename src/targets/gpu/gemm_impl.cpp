/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <rocblas/rocblas.h>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/permutation.hpp>

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
    case shape::tuple_type:
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
}

void blas_shape(const shape& s)
{
    if(s.lens().size() < 2)
        return;
    if(std::none_of(s.strides().end() - 2, s.strides().end(), [&](auto i) { return i == 1; }))
        MIGRAPHX_THROW("GPU_GEMM: needs to have one matrix stride as 1");
    if(s.lens().size() < 3)
        return;
    shape batch_shape{s.type(),
                      {s.lens().begin(), s.lens().end() - 2},
                      {s.strides().begin(), s.strides().end() - 2}};
    auto batch_shapes = reduce_dims({batch_shape});
    if(batch_shapes.front().lens().size() != 1)
        MIGRAPHX_THROW("GPU_GEMM: Batch dimension is not collapsible");
}

shape transpose_batch(const shape& s, unsigned trans_batch)
{
    if(trans_batch == 0)
        return s;
    if(s.lens().size() < 3)
        return s;
    auto batch = s.lens().size() - 3;
    std::vector<int64_t> perm(s.lens().size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[batch], perm[batch + trans_batch]);
    return shape::from_permutation(s.type(), s.lens(), perm);
}

template <class R, class... Ts, class... Us>
R rocblas_invoke(R (*f)(Ts...), Us... xs)
{
    if constexpr(sizeof...(Ts) == sizeof...(Us))
        return f(xs...);
    else
        return f(xs..., nullptr, nullptr);
}

static bool is_transposed(const shape& s)
{
    if(not s.transposed())
        return false;
    return s.strides().back() != 1;
}

static rocblas_int get_batch_stride(const argument& a)
{
    return a.get_shape().strides()[a.get_shape().strides().size() - 3];
}

template <class T>
void gemm_impl(context& ctx,
               const shape& output_shape,
               const std::vector<argument>& args,
               T alpha,
               T beta,
               bool int8_x4_format,
               bool compute_fp32)
{
    const bool is_3inputs = (args.size() == 4);
    if(not is_3inputs)
    {
        beta = 0;
    }

    bool transa     = is_transposed(args[0].get_shape());
    bool transb     = is_transposed(args[1].get_shape());
    auto n_dim      = output_shape.lens().size();
    auto dim_1      = n_dim - 1;
    auto dim_0      = n_dim - 2;
    rocblas_int lda = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
    rocblas_int ldb = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
    rocblas_int ldc = args[2].get_shape().strides()[dim_0];
    rocblas_int ldd = is_3inputs ? args[3].get_shape().strides()[dim_0] : ldc;

    rocblas_datatype arg_type = get_type(args[0].get_shape().type());
    auto output_type          = arg_type;
    if(output_type == rocblas_datatype_i8_r)
    {
        output_type = rocblas_datatype_i32_r;
    }
    auto compute_type = output_type;
    if(compute_fp32)
    {
        if(arg_type == rocblas_datatype_f16_r)
            compute_type = rocblas_datatype_f32_r;
    }

    rocblas_gemm_flags flag = rocblas_gemm_flags_none;
#if ROCBLAS_VERSION_MAJOR < 3
    if(int8_x4_format)
        flag = rocblas_gemm_flags_pack_int8x4;
#endif

    auto a_lens = args[0].get_shape().lens();
    auto b_lens = args[1].get_shape().lens();
    output_shape.visit_type([&](auto as) {
        auto alpha_r = as(alpha);
        auto beta_r  = as(beta);

        // use void pointer to select different data type if using fp32 mode
        void* alpha_v = &alpha_r;
        void* beta_v  = &beta_r;

        if(compute_fp32)
        {
            alpha_v = &alpha;
            beta_v  = &beta;
        }

        auto out_lens   = output_shape.lens();
        rocblas_int m   = out_lens[dim_0];
        rocblas_int n   = out_lens[dim_1];
        rocblas_int k   = args[0].get_shape().lens()[dim_1];
        auto to_pointer = [&](auto&& arg) { return as.from(arg.data()); };
        if(args[0].get_shape().type() == shape::int8_type and (k % 4) != 0 and int8_x4_format)
        {
            MIGRAPHX_THROW("ROCBLAS_GEMM: k size of int8 type input must be mutlple of 4!");
        }

        auto num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        if(num_matrices == 1 or (num_matrices > 1 and get_batch_stride(args[1]) == 0))
        {
            // If the batch dimension of B is broadcasted, then we can
            // multiply m by the batch_size and use rocblas_gemm_ex
            // instead of rocblas_gemm_strided_batched_ex.
            m *= num_matrices;

            // the rocblas_gemm API handles inputs and output matrices as
            // column-major format. When doing a C = A * B, we actually do
            // C^T = (B^T) * (A^T). That is the reason we input args[1] as
            // A and args[0] as B in calling the rocblas_gemm.
            rocblas_invoke(&rocblas_gemm_ex,
                           ctx.get_stream().get_rocblas(),
                           transb ? rocblas_operation_transpose : rocblas_operation_none,
                           transa ? rocblas_operation_transpose : rocblas_operation_none,
                           n,
                           m,
                           k,
                           alpha_v,
                           to_pointer(args.at(1)),
                           arg_type,
                           ldb,
                           to_pointer(args.at(0)),
                           arg_type,
                           lda,
                           beta_v,
                           to_pointer(args[2]),
                           output_type,
                           ldc,
                           is_3inputs ? to_pointer(args[3]) : to_pointer(args[2]),
                           output_type,
                           ldd,
                           compute_type,
                           rocblas_gemm_algo_standard,
                           0,
                           flag);
        }
        else
        {
            auto a_stride = get_batch_stride(args[0]);
            auto b_stride = get_batch_stride(args[1]);
            auto c_stride = get_batch_stride(args[2]);
            auto d_stride = is_3inputs ? get_batch_stride(args[3]) : c_stride;
            rocblas_invoke(&rocblas_gemm_strided_batched_ex,
                           ctx.get_stream().get_rocblas(),
                           transb ? rocblas_operation_transpose : rocblas_operation_none,
                           transa ? rocblas_operation_transpose : rocblas_operation_none,
                           n,
                           m,
                           k,
                           alpha_v,
                           to_pointer(args.at(1)),
                           arg_type,
                           ldb,
                           b_stride,
                           to_pointer(args.at(0)),
                           arg_type,
                           lda,
                           a_stride,
                           beta_v,
                           to_pointer(args[2]),
                           output_type,
                           ldc,
                           c_stride,
                           is_3inputs ? to_pointer(args[3]) : to_pointer(args[2]),
                           output_type,
                           ldd,
                           d_stride,
                           num_matrices,
                           compute_type,
                           rocblas_gemm_algo_standard,
                           0,
                           flag);
        }
    });
}

void gemm(context& ctx,
          const shape& output_shape,
          const std::vector<argument>& args,
          float alpha,
          float beta,
          bool int8_x4_format,
          bool compute_fp32)
{
    gemm_impl(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
}

void gemm(context& ctx,
          const shape& output_shape,
          const std::vector<argument>& args,
          int32_t alpha,
          int32_t beta,
          bool int8_x4_format,
          bool compute_fp32)
{
    gemm_impl(ctx, output_shape, args, alpha, beta, int8_x4_format, compute_fp32);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
