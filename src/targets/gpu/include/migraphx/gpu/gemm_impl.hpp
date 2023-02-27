/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iterator>
#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/time.hpp>

using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

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

static rocblas_int get_batch_stride(const shape& a) { return a.strides()[a.strides().size() - 3]; }

/**
 * Returns results of rocblas_status_success, rocblas_status_perf_degraded,
 * or rocblas_status_invalid_value.  Caller
 * is expected to check for invalid index.  Any other result causes an exception.
 *
 */
template <class F, class Pack, class... Ts>
auto rocblas_invoke(F f, Pack p, Ts... xs)
{
    return p([=](auto... ws) {
        auto status = f(ws..., xs...);
        if(status != rocblas_status_success and status != rocblas_status_invalid_value)
        {
            if(status == rocblas_status_perf_degraded)
            {
                std::cerr << "WARNING: degraded perf. in rocBLAS call" << std::endl;
            }
            else
                MIGRAPHX_THROW("rocblas_invoke: rocBLAS call failed with status " +
                               std::to_string(status));
        }
        return status;
    });
}

template <typename T>
struct gemm_impl
{
    gemm_impl(const shape& output_shape,
              const std::vector<shape>& input_shapes,
              T alpha_param,
              T beta_param,
              bool int8_x4_format,
              bool compute_fp32_flag)
        : alpha(alpha_param), beta(beta_param), compute_fp32(compute_fp32_flag)
    {
        is_3inputs = (input_shapes.size() == 4);
        if(not is_3inputs)
        {
            beta = 0;
        }

        transa     = is_transposed(input_shapes[0]);
        transb     = is_transposed(input_shapes[1]);
        auto n_dim = output_shape.lens().size();
        auto dim_0 = n_dim - 2;
        auto dim_1 = n_dim - 1;
        // Leading dimensions of matrices
        lda = input_shapes[0].strides()[transa ? dim_1 : dim_0];
        ldb = input_shapes[1].strides()[transb ? dim_1 : dim_0];
        ldc = input_shapes[2].strides()[dim_0];
        ldd = is_3inputs ? input_shapes[3].strides()[dim_0] : ldc;

        arg_type    = get_type(input_shapes[0].type());
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

        int8_flag = int8_x4_format ? rocblas_gemm_flags_pack_int8x4 : rocblas_gemm_flags_none;

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

        auto a_lens = input_shapes[0].lens();
        auto b_lens = input_shapes[1].lens();

        auto out_lens = output_shape.lens();
        m             = out_lens[dim_0];
        n             = out_lens[dim_1];
        k             = input_shapes[0].lens()[dim_1];
        if(input_shapes[0].type() == shape::int8_type and (k % 4) != 0 and int8_x4_format)
        {
            MIGRAPHX_THROW("ROCBLAS_GEMM: k size of int8 type input must be mutlple of 4!");
        }
        a_stride     = get_batch_stride(input_shapes[0]);
        b_stride     = get_batch_stride(input_shapes[1]);
        c_stride     = get_batch_stride(input_shapes[2]);
        d_stride     = is_3inputs ? get_batch_stride(input_shapes[3]) : c_stride;
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

    auto create_gemm_args(context& ctx,
                          const std::vector<argument>& args,
                          rocblas_gemm_flags flag,
                          int32_t solution_idx = 0) const
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
                    rocblas_gemm_algo_solution_index,
                    solution_idx,
                    flag);
    }

    auto create_strided_batched_gemm_args(context& ctx,
                                          const std::vector<argument>& args,
                                          rocblas_gemm_flags flag,
                                          int32_t solution_idx = 0) const
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
                    rocblas_gemm_algo_solution_index,
                    solution_idx,
                    flag);
    }

    void run(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx = 0) const
    {

        if(strided_batched)
        {
            auto gemm_args =
                create_strided_batched_gemm_args(ctx, input_args, int8_flag, solution_idx);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex, gemm_args);
        }
        else
        {
            auto gemm_args = create_gemm_args(ctx, input_args, int8_flag, solution_idx);
            rocblas_invoke(&rocblas_gemm_ex, gemm_args);
        }
    }

    auto validate(context& ctx, const std::vector<shape>& input_shapes, int32_t solution_idx) const
    {
        std::vector<argument> input_args;
        std::transform(input_shapes.begin(),
                       input_shapes.end(),
                       std::back_inserter(input_args),
                       [](const shape& x) { return to_gpu(generate_argument(x)); });

        return validate(ctx, input_args, solution_idx);
    }

    /**
     * Checks a particular solution for validity by running it with the flag
     * rocblas_gemm_flags_check_solution_index, and returns the result
     */
    auto validate(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx) const
    {
        if(strided_batched)
        {
            auto gemm_args = create_strided_batched_gemm_args(
                ctx, input_args, rocblas_gemm_flags_check_solution_index, solution_idx);
            return rocblas_invoke(&rocblas_gemm_strided_batched_ex, gemm_args);
        }
        else
        {
            auto gemm_args = create_gemm_args(
                ctx, input_args, rocblas_gemm_flags_check_solution_index, solution_idx);
            return rocblas_invoke(&rocblas_gemm_ex, gemm_args);
        }
    }

    void print_args() const
    {
        std::cout << "trans: " << transa << " transb: " << transb << "\n";
        std::cout << "m: " << m << " n: " << n << " k: " << k << "\n";
        std::cout << "alpha: " << alpha << " beta: " << beta << "\n";
        std::cout << "lda : " << lda << " ldb: " << ldb << " ldc: " << ldc << " ldd: " << ldd
                  << "\n";
        std::cout << "strided_batched: " << strided_batched << "\n";
        std::cout << "astride: " << a_stride << " b_stride: " << b_stride
                  << " c_stride: " << c_stride << " d_stride: " << d_stride << "\n";
        std::cout << "arg type : " << arg_type << " compute_type: " << compute_type
                  << " output_type: " << output_type << "\n";
        std::cout << "int8_flag: " << int8_flag << "\n";
    }

    auto create_strided_batched_gemm_get_solutions_args(context& ctx,
                                                        const std::vector<argument>& args,
                                                        rocblas_int* list,
                                                        rocblas_int* list_size) const
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
                    rocblas_gemm_algo_solution_index,
                    int8_flag,
                    list,
                    list_size);
    }

    auto create_gemm_ex_get_solutions_args(context& ctx,
                                           const std::vector<argument>& args,
                                           rocblas_int* list,
                                           rocblas_int* list_size) const
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
                    rocblas_gemm_algo_solution_index,
                    int8_flag,
                    list,
                    list_size);
    }

    int tune(context& ctx, const std::vector<shape>& input_shapes) const
    {
        std::vector<argument> input_args;
        std::transform(input_shapes.begin(),
                       input_shapes.end(),
                       std::back_inserter(input_args),
                       [](const shape& x) { return to_gpu(generate_argument(x)); });

        // Find out how many solutions there are
        rocblas_int list_size = 0;
        std::vector<rocblas_int> solution_indices;
        if(strided_batched)
        {
            auto arg_list = create_strided_batched_gemm_get_solutions_args(
                ctx, input_args, nullptr, &list_size);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex_get_solutions, arg_list);
            solution_indices.resize(list_size);
            auto arg_list_solutions = create_strided_batched_gemm_get_solutions_args(
                ctx, input_args, solution_indices.data(), &list_size);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex_get_solutions, arg_list_solutions);
        }
        else
        {
            auto arg_list = create_gemm_ex_get_solutions_args(ctx, input_args, nullptr, &list_size);
            rocblas_invoke(&rocblas_gemm_ex_get_solutions, arg_list);
            solution_indices.resize(list_size);
            auto arg_list_solutions = create_gemm_ex_get_solutions_args(
                ctx, input_args, solution_indices.data(), &list_size);
            rocblas_invoke(&rocblas_gemm_ex_get_solutions, arg_list_solutions);
        }

        double bestTime = std::numeric_limits<double>::max();
        // Initialize to default solution index
        rocblas_int bestSol = 0;
        for(auto sol : solution_indices)
        {
            for(rocblas_int cc = 0; cc < cold_calls; ++cc)
            {
                run(ctx, input_args, sol);
            }
            double host_time = 0.0;
            auto run_func    = [&]() {
                run(ctx, input_args, sol);
                ctx.finish();
            };
            for(rocblas_int hc = 0; hc < hot_calls; ++hc)
            {
                ctx.finish();
                host_time += time<microseconds>(run_func);
            }
            // todo:  Measured time dropped from 20 us to about 6.7 us when I raised hot_calls from
            // 1 to 11. The higher the hot_calls value, the faster per-call time up to at least 25,
            // and increasing cold_calls makes little or no difference.  Why?
            host_time /= hot_calls;
            std::cout << "Solution index " << sol << ": " << host_time << " us" << std::endl;

            // track current best
            if(host_time < bestTime)
            {
                printf(" current best index  %d\n", sol);
                bestSol  = sol;
                bestTime = host_time;
            }
        }
        std::cout << "Winner: " << bestSol << " in " << bestTime << " us" << std::endl;
        return bestSol;
    }

    private:
    size_t num_matrices;
    rocblas_int m, n, k;
    bool transa, transb;
    T alpha, beta;
    void* alpha_v = nullptr;
    void* beta_v  = nullptr;
    rocblas_gemm_flags int8_flag;
    rocblas_int lda, ldb, ldc, ldd;
    rocblas_int a_stride, b_stride, c_stride, d_stride;
    rocblas_datatype compute_type, arg_type, output_type;
    bool strided_batched = true, is_3inputs = true, compute_fp32 = true;
    uint32_t flags = 0; //  optional gemm flags.
    // tuning meta parameters
    rocblas_int cold_calls = 4;
    rocblas_int hot_calls  = 100;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
