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
#ifndef MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GEMM_IMPL_HPP

#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/reduce_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

inline rocblas_datatype get_type(shape::type_t type)
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

inline void blas_shape(const shape& s)
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

inline shape transpose_batch(const shape& s, unsigned trans_batch)
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

template <class F, class Pack, class... Ts>
auto rocblas_invoke(F f, Pack p, Ts... xs)
{
    return p([=](auto... ws) { return f(ws..., xs...); });
}

template <typename T>
struct gemm_impl
{
    gemm_impl(const shape& output_shape,
              const std::vector<argument>& args,
              T alpha_param,
              T beta_param,
              bool int8_x4_format,
              bool compute_fp32_flag)
        : alpha(alpha_param), beta(beta_param), compute_fp32(compute_fp32_flag)
    {
        is_3inputs = (args.size() == 4);
        if(not is_3inputs)
        {
            beta = 0;
        }

        transa     = is_transposed(args[0].get_shape());
        transb     = is_transposed(args[1].get_shape());
        auto n_dim = output_shape.lens().size();
        auto dim_0 = n_dim - 2;
        auto dim_1 = n_dim - 1;
        lda        = args[0].get_shape().strides()[transa ? dim_1 : dim_0];
        ldb        = args[1].get_shape().strides()[transb ? dim_1 : dim_0];
        ldc        = args[2].get_shape().strides()[dim_0];
        ldd        = is_3inputs ? args[3].get_shape().strides()[dim_0] : ldc;

        arg_type    = get_type(args[0].get_shape().type());
        output_type = arg_type;
        if(output_type == rocblas_datatype_i8_r)
        {
            output_type = rocblas_datatype_i32_r;
        }
        compute_type = output_type;
        if(compute_fp32)
        {
            if(arg_type == rocblas_datatype_f16_r)
                compute_type = rocblas_datatype_f32_r;
        }

        flag = int8_x4_format ? rocblas_gemm_flags_pack_int8x4 : rocblas_gemm_flags_none;

        // use void pointer to select different data type if using fp32 mode
        output_shape.visit_type([&](auto as) {
            auto alpha_r = as(alpha);
            auto beta_r  = as(beta);
            if(compute_fp32)
            {
                alpha_v = &alpha;
                beta_v  = &beta;
            }
            else
            {
                alpha_v = &alpha_r;
                beta_v  = &beta_r;
            }
        });

        auto a_lens = args[0].get_shape().lens();
        auto b_lens = args[1].get_shape().lens();

        auto out_lens = output_shape.lens();
        m             = out_lens[dim_0];
        n             = out_lens[dim_1];
        k             = args[0].get_shape().lens()[dim_1];
        if(args[0].get_shape().type() == shape::int8_type and (k % 4) != 0 and int8_x4_format)
        {
            MIGRAPHX_THROW("ROCBLAS_GEMM: k size of int8 type input must be mutlple of 4!");
        }
        a_stride     = get_batch_stride(args[0]);
        b_stride     = get_batch_stride(args[1]);
        c_stride     = get_batch_stride(args[2]);
        d_stride     = is_3inputs ? get_batch_stride(args[3]) : c_stride;
        num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());

        if(num_matrices == 1 or (num_matrices > 1 and b_stride == 0))
        {
            // If the batch dimension of B is broadcasted, then we can
            // multiply m by the batch_size and use rocblas_gemm_ex
            // instead of rocblas_gemm_strided_batched_ex.
            m *= num_matrices;
            strided_batched = false;
        }
    }

    auto create_gemm_args(context& ctx, const std::vector<argument>& args)
    {
        // the rocblas_gemm API handles inputs and output matrices as
        // column-major format. When doing a C = A * B, we actually do
        // C^T = (B^T) * (A^T). That is the reason we input args[1] as
        // A and args[0] as B in calling the rocblas_gemm.
        return pack(ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    alpha_v,
                    args[1].data(),
                    arg_type,
                    ldb,
                    args[0].data(),
                    arg_type,
                    lda,
                    beta_v,
                    args[2].data(),
                    output_type,
                    ldc,
                    is_3inputs ? args[3].data() : args[2].data(),
                    output_type,
                    ldd,
                    compute_type,
                    rocblas_gemm_algo_standard,
                    solution_idx,
                    flag);
    }

    auto create_strided_batched_gemm_args(context& ctx, const std::vector<argument>& args)
    {
        return pack(ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    alpha_v,
                    args[1].data(),
                    arg_type,
                    ldb,
                    b_stride,
                    args[0].data(),
                    arg_type,
                    lda,
                    a_stride,
                    beta_v,
                    args[2].data(),
                    output_type,
                    ldc,
                    c_stride,
                    is_3inputs ? args[3].data() : args[2].data(),
                    output_type,
                    ldd,
                    d_stride,
                    num_matrices,
                    compute_type,
                    rocblas_gemm_algo_standard,
                    solution_idx,
                    flag);
    }

    void run(context& ctx, const std::vector<argument>& input_args)
    {

        if(strided_batched)
        {
            auto gemm_args = create_strided_batched_gemm_args(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex, gemm_args);
        }
        else
        {
            auto gemm_args = create_gemm_args(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_ex, gemm_args);
        }
    }

    void print_args() const
    {
        std::cout << "trans: " << transa << " transb: " << transb << "\n";
        std::cout << "m: " << m << " n: " << n << " k: " << k << "\n";
        std::cout << "alpha: " << alpha << " beta: " << beta << "\n";
        std::cout << "lda : " << lda << " ldb: " << ldb << " ldc: " << ldc << " ldd: " << ldd
                  << "\n";
        std::cout << "strided_batched: " << strided_batched << " is_3inputs: " << is_3inputs
                  << "\n";
        std::cout << "astride: " << a_stride << " b_stride: " << b_stride
                  << " c_stride: " << c_stride << " d_stride: " << d_stride << "\n";
        std::cout << "arg type : " << arg_type << " compute_type: " << compute_type
                  << " output_type: " << output_type << "\n";
        std::cout << "flag: " << flag << "\n";
        std::cout << "solution_idx: " << solution_idx << "\n";
    }

    private:
    size_t num_matrices;
    rocblas_int m, n, k;
    bool transa, transb;
    T alpha, beta;
    void* alpha_v = nullptr;
    void* beta_v  = nullptr;
    rocblas_gemm_flags flag;
    rocblas_int lda, ldb, ldc, ldd;
    rocblas_int solution_idx = 0;
    rocblas_int a_stride, b_stride, c_stride, d_stride;
    rocblas_datatype compute_type, arg_type, output_type;
    bool strided_batched = true, is_3inputs = false, compute_fp32 = true;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
