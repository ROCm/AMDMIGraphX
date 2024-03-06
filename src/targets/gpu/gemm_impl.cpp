/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocblas/internal/rocblas-types.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/gemm_impl.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>
#include <type_traits>

using microseconds = std::chrono::duration<double, std::micro>;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hipblaslt_args
{
    hipblasLtHandle_t handle;
    hipblasLtMatmulPreference_t preference;
};

/*
Regular rocBLAS API takes compute_type as `rocblas_datatype` enum value v/s "ex3" BETA API takes it
as `rocblas_computetype` enum value. `rb_compute_type` is faciliator to implictly cast integer enum
value to required type that can be used inside `common_args` generator.
*/
struct rb_compute_type
{
    int type = 0;
    rb_compute_type(rocblas_datatype t) : type(static_cast<int>(t)) {}
    rb_compute_type(rocblas_computetype t) : type(static_cast<int>(t)) {}
    operator rocblas_datatype() const { return static_cast<rocblas_datatype>(type); }
    operator rocblas_computetype() const { return static_cast<rocblas_computetype>(type); }
};

// Convert rocBLAS datatypes to equivalent Migraphx data types
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
    case shape::fp8e4m3fnuz_type: return rocblas_datatype_f8_r;
    case shape::tuple_type:
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("ROCBLAS_GEMM: data type not supported!");
}

// Convert hipBLAS datatypes to equivalent Migraphx data types
hipblasltDatatype_t get_type_hipblas(shape::type_t type)
{
    switch(type)
    {
    case shape::double_type: return HIPBLASLT_R_64F;
    case shape::float_type: return HIPBLASLT_R_32F;
    case shape::half_type: return HIPBLASLT_R_16F;
    case shape::int8_type: return HIPBLASLT_R_8I;
    case shape::uint8_type: return HIPBLASLT_R_8U;
    case shape::int32_type: return HIPBLASLT_R_32I;
    case shape::uint32_type: return HIPBLASLT_R_32U;
    case shape::fp8e4m3fnuz_type: return HIPBLASLT_R_8F_E4M3;
    case shape::tuple_type:
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("HIPBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("HIPBLAS_GEMM: data type not supported!");
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

/**
 * Returns results of rocblas_status_success, rocblas_status_perf_degraded,
 * or rocblas_status_invalid_value.  Caller
 * is expected to check for invalid index.  Any other result causes an exception.
 *
 */
template <class F, class Pack, class... Ts>
auto hipblaslt_invoke(F f, Pack p, Ts... xs)
{
    return p([=](auto... ws) {
        auto status = f(ws..., xs...);
        if(status != HIPBLAS_STATUS_SUCCESS)
        //if(status != HIPBLAS_STATUS_SUCCESS and status != HIPBLAS_STATUS_INVALID_VALUE)
        {
//            if(status == rocblas_status_perf_degraded)
//            {
//                std::cerr << "WARNING: degraded perf. in rocBLAS call" << std::endl;
//            }
//            else
                MIGRAPHX_THROW("hipblaslt_invoke: hipBlasLt call failed with status " +
                               std::to_string(status));
        }
        return status;
    });
}

static bool is_transposed(const shape& s) { return s.transposed() and s.strides().back() != 1; }

static rocblas_int get_batch_stride(const shape& s)
{
    // This value is not needed for non-strided inputs
    if(s.strides().size() < 3)
        return 0;
    else
        return s.strides()[s.strides().size() - 3];
}

/**
 * Wrapper for multiple rocBLAS calls.  The constructor creates parameters for
 * these calls based on data shapes and other values contained in the associated
 * instruction and operation.
 *
 * The template parameter T is not the type of the matrix data but of the weighting
 * coefficients alpha and beta (these are float in rocBLAS internals)
 */
template <typename T>
struct gemm_impl
{
    gemm_impl(const shape& output_shape,
              const std::vector<shape>& input_shapes,
              T alpha_param,
              T beta_param,
              bool compute_fp32_flag)
        : solution(),
          alpha(alpha_param),
          beta(beta_param),
          is_3inputs(input_shapes.size() == 4),
          compute_fp32(compute_fp32_flag)
    {
        if(not is_3inputs)
        {
            beta = 0;
        }

        // Create lambdas that will cast alpha, beta to the output shape's type
        // and retain the values being pointed to
        output_shape.visit_type([&](auto as) {
            auto alpha_r = as(alpha);
            auto beta_r  = as(beta);
            if(compute_fp32)
            {
                get_alpha = [=] { return &alpha; };
                get_beta  = [=] { return &beta; };
            }
            else
            {
                get_alpha = [=] { return &alpha_r; };
                get_beta  = [=] { return &beta_r; };
            }
        });

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
        output_type = get_type(input_shapes[2].type());
        if(output_type == rocblas_datatype_i8_r or output_type == rocblas_datatype_u8_r)
        {
            output_type = rocblas_datatype_i32_r;
        }
        compute_type = rb_compute_type{output_type};
        if(compute_fp32)
        {
            if(arg_type == rocblas_datatype_f16_r)
                compute_type = rocblas_datatype_f32_r;
        }
        if(arg_type == rocblas_datatype_f8_r)
        {
            assert(get_type(input_shapes[1].type()) == rocblas_datatype_f8_r);
            compute_type = rocblas_compute_type_f32;
        }

        auto a_lens = input_shapes[0].lens();
        auto b_lens = input_shapes[1].lens();

        auto out_lens = output_shape.lens();
        m             = out_lens[dim_0];
        n             = out_lens[dim_1];
        k             = input_shapes[0].lens()[dim_1];

        a_stride     = get_batch_stride(input_shapes[0]);
        b_stride     = get_batch_stride(input_shapes[1]);
        c_stride     = get_batch_stride(input_shapes[2]);
        d_stride     = is_3inputs ? get_batch_stride(input_shapes[3]) : c_stride;
        num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        strided_batched = num_matrices > 1;
        bool rocblas_fallback = strided_batched and b_stride == 0 and input_shapes[0].standard();
        if(rocblas_fallback)
        {
            // If the batch dimension of B is broadcasted, then we can
            // multiply m by the batch_size and use rocblas_gemm_ex
            // instead of rocblas_gemm_strided_batched_ex.
            m *= num_matrices;
            strided_batched = false;
        }

        // hipblaslt
        dtype = get_type_hipblas(input_shapes[0].type());

        // casting from rocblas_int which is int64_t
        const uint64_t m_ = static_cast<uint64_t>(n);
        const uint64_t n_ = static_cast<uint64_t>(rocblas_fallback ? m/num_matrices : m);
        const uint64_t k_ = static_cast<uint64_t>(k);

        op_A = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        op_B = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;

        const int lda_ = static_cast<int>(ldb);
        const int ldb_ = static_cast<int>(lda);
        const int ldc_ = static_cast<int>(ldc);
        const int ldd_ = static_cast<int>(ldd);

        if(op_A == HIPBLAS_OP_N)
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype, m_, k_, lda_));
        }
        else
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matA, dtype, k_, m_, lda_));
        }
        if(op_B == HIPBLAS_OP_N)
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype, k_, n_, ldb_));
        }
        else
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matB, dtype, n_, k_, ldb_));
        }
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matC, dtype, m_, n_, ldc_));
        if(is_3inputs)
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutCreate(&matD, dtype, m_, n_, ldd_));
        }

        if(num_matrices > 1)
        {
            const int64_t a_stride_ = static_cast<int64_t>(b_stride);
            const int64_t b_stride_ = static_cast<int64_t>(a_stride);
            const int64_t c_stride_ = static_cast<int64_t>(c_stride);
            const int64_t d_stride_ = static_cast<int64_t>(d_stride);

            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices)));
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices)));
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices)));

            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matA, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &a_stride_, sizeof(a_stride_)));
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matB, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &b_stride_, sizeof(b_stride_)));
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                matC, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &c_stride_, sizeof(c_stride_)));


            if(is_3inputs)
            {
                CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                    matD, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices)));
                CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutSetAttribute(
                    matD, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &d_stride_, sizeof(d_stride_)));
            }
        }

        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescCreate(&hipblaslt_desc, HIPBLASLT_COMPUTE_F32, HIPBLASLT_R_32F));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));
    }

    ~gemm_impl()
    {
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescDestroy(hipblaslt_desc));
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matA));
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matB));
        CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matC));
        if(is_3inputs)
        {
            CHECK_HIPBLAS_ERROR(hipblasLtMatrixLayoutDestroy(matD));
        }
    }

    struct solution
    {
        solution() : handle(nullptr), preference(nullptr) {}

        void init(context& ctx)
        {
            if (handle == nullptr)
            {
                handle = ctx.get_stream().get_hipblaslt();
                preference = ctx.get_stream().get_hipblaslt_preference();
            }
        }

        // Suppose neither rocblasa or hipblaslt index uses bits up to 30
        const int BLAS_BIT = 30;
        bool is_rocblas_idx(int32_t idx) { return ((idx >> BLAS_BIT) & 1); }
        int32_t from_mgx_to_blas_idx(int32_t idx) { return (idx & ~(1 << BLAS_BIT)); }
        int32_t from_blas_to_mgx_idx(int32_t idx, bool is_rocblas) { return is_rocblas? idx : (idx | (1 << BLAS_BIT)); }

        auto &get_result(context& ctx, gemm_impl& gemm, int32_t idx)
        {
            init(ctx);

            if(idx == 0)
            {
                // use default solution
                const int n_sol = 1;
                int returnedAlgoCount;
                heuristicResult.resize(n_sol);
                CHECK_HIPBLAS_ERROR(hipblasLtMatmulAlgoGetHeuristic(
                    handle, gemm.hipblaslt_desc, gemm.matA, gemm.matB, gemm.matC, gemm.matC,
                    preference, n_sol, heuristicResult.data(), &returnedAlgoCount));
                if(returnedAlgoCount != n_sol)
                {
                    std::cout << "less solution found! request: " << n_sol
                              << ", found: " << returnedAlgoCount << std::endl;
                }
            }
            else
            {
                // query for the solutions. 1st as the best.
                std::vector<int> algoIndex(1);
                algoIndex[0] = idx;
                CHECK_HIPBLAS_ERROR(hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResult));
            }
            return heuristicResult;
        }

        private:

        hipblasLtHandle_t handle;
        hipblasLtMatmulPreference_t preference;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    } solution;

#define IS_ROCBLAS(x)   solution.is_rocblas_idx(x)
#define AS_BLAS_IDX(x)  solution.from_mgx_to_blas_idx(x)
#define AS_MGX_IDX(x,b) solution.from_blas_to_mgx_idx(x,b)

    void run(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx = 0) //const
    {
        if(IS_ROCBLAS(solution_idx))
        {
            solution_idx = AS_BLAS_IDX(solution_idx);
#ifdef MIGRAPHX_USE_ROCBLAS_FP8_API
            if(rocblas_fp8_available() and
               std::any_of(input_args.begin(), input_args.end(), [](const auto i) {
                   return i.get_shape().type() == migraphx::shape::fp8e4m3fnuz_type;
               }))
            {
                if(strided_batched)
                {
                    auto common_args = create_strided_batched_args_common(ctx, input_args);
                    rocblas_invoke(&rocblas_gemm_strided_batched_ex3,
                                   common_args,
                                   rocblas_gemm_algo_standard,
                                   solution_idx,
                                   gemm_flags);
                }
                else
                {
                    auto common_args = create_gemm_ex_args_common(ctx, input_args);
                    rocblas_invoke(&rocblas_gemm_ex3,
                                   common_args,
                                   rocblas_gemm_algo_standard,
                                   solution_idx,
                                gemm_flags);
                }
            }
            else
#endif
            {
                if(strided_batched)
                {
                    auto common_args = create_strided_batched_args_common(ctx, input_args);
                    rocblas_invoke(&rocblas_gemm_strided_batched_ex,
                                   common_args,
                                   rocblas_gemm_algo_solution_index,
                                   solution_idx,
                                   gemm_flags);
                }
                else
                {
                    auto common_args = create_gemm_ex_args_common(ctx, input_args);
                    rocblas_invoke(&rocblas_gemm_ex,
                                   common_args,
                                   rocblas_gemm_algo_solution_index,
                                   solution_idx,
                                   gemm_flags);
                }
            }
        }
        else
        {
            solution_idx = AS_BLAS_IDX(solution_idx);
            auto common_args = create_hipblaslt_args_common(ctx, input_args, solution_idx);
            hipblaslt_invoke(&hipblasLtMatmul, common_args);
        }
    }

#ifdef MIGRAPHX_USE_ROCBLAS_TUNING_API
    auto validate(context& ctx, const std::vector<shape>& input_shapes, int32_t solution_idx)// const
    {
        // Create dummy arguments for the shapes, and call the overloaded method
        std::vector<argument> input_args;
        std::transform(input_shapes.begin(),
                       input_shapes.end(),
                       std::back_inserter(input_args),
                       [](const shape& x) { return to_gpu(generate_argument(x)); });

        return validate(ctx, input_args, solution_idx);
    }

    /**
     * Checks a particular solution for validity by running it with the flag
     * rocblas_gemm_flags_check_solution_index (could be invalid if this model was
     * tuned with a different rocBLAS version)
     *
     * @return Returns either solution_idx if valid, or else the default value 0
     * if not.  The default does not mean list index 0, but tells the picker
     * to choose a solution.
     */
    int32_t
    validate(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx)// const
    {
        if(IS_ROCBLAS(solution_idx))
        {
            rocblas_status_ check_valid(rocblas_status_success);
            solution_idx = AS_BLAS_IDX(solution_idx);
            if(strided_batched)
            {
                auto common_args = create_strided_batched_args_common(ctx, input_args);
                check_valid      = rocblas_invoke(&rocblas_gemm_strided_batched_ex,
                                             common_args,
                                             rocblas_gemm_algo_solution_index,
                                             solution_idx,
                                             rocblas_gemm_flags_check_solution_index);
            }
            else
            {
                auto common_args = create_gemm_ex_args_common(ctx, input_args);
                check_valid      = rocblas_invoke(&rocblas_gemm_ex,
                                             common_args,
                                             rocblas_gemm_algo_solution_index,
                                             solution_idx,
                                             rocblas_gemm_flags_check_solution_index);
            }

            if(check_valid == rocblas_status_invalid_value)
            {
                std::cerr << "WARNING:  tuned solution is invalid; reverting to default" << std::endl;
                return 0;
            }
            return AS_MGX_IDX(solution_idx, true);
        }
        else
        {
            hipblasStatus_t check_valid(HIPBLAS_STATUS_SUCCESS);
            solution_idx = AS_BLAS_IDX(solution_idx);
            auto common_args = create_hipblaslt_args_common(ctx, input_args, solution_idx);
            check_valid = hipblaslt_invoke(&hipblasLtMatmul, common_args);
            if(check_valid == HIPBLAS_STATUS_SUCCESS)
            {
                std::cerr << "WARNING:  tuned solution is invalid; reverting to default" << std::endl;
                return 0;
            }

            return AS_MGX_IDX(solution_idx, false);
        }
    }
#endif

    /**
     * Helper method to create that subset of a long rocBLAS argument list that is common
     * to multiple "...strided_batched..." calls.
     *
     * The rocblas_gemm API handles inputs and output matrices as
     *  column-major format. When doing a C = A * B, we actually do
     *  C^T = (B^T) * (A^T). That is the reason we input args[1] as
     *   A and args[0] as B in calling the rocblas_gemm.
     *
     */
    auto create_strided_batched_args_common(context& ctx, const std::vector<argument>& args) const
    {
        return pack(ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    get_alpha(),
                    args[1].data(),
                    arg_type,
                    ldb,
                    b_stride,
                    args[0].data(),
                    arg_type,
                    lda,
                    a_stride,
                    get_beta(),
                    args[2].data(),
                    output_type,
                    ldc,
                    c_stride,
                    is_3inputs ? args[3].data() : args[2].data(),
                    output_type,
                    ldd,
                    d_stride,
                    num_matrices,
                    compute_type);
    }
    /**
     * Helper method to create that subset of a long rocBLAS argument list that is common
     * to multiple "gemm_ex..." calls.
     *
     * The rocblas_gemm API handles inputs and output matrices as
     *  column-major format. When doing a C = A * B, we actually do
     *   C^T = (B^T) * (A^T). That is the reason we input args[1] as
     *   A and args[0] as B in calling the rocblas_gemm.
     *
     * */
    auto create_gemm_ex_args_common(context& ctx, const std::vector<argument>& args) const
    {
        return pack(ctx.get_stream().get_rocblas(),
                    transb ? rocblas_operation_transpose : rocblas_operation_none,
                    transa ? rocblas_operation_transpose : rocblas_operation_none,
                    n,
                    m,
                    k,
                    get_alpha(),
                    args[1].data(),
                    arg_type,
                    ldb,
                    args[0].data(),
                    arg_type,
                    lda,
                    get_beta(),
                    args[2].data(),
                    output_type,
                    ldc,
                    is_3inputs ? args[3].data() : args[2].data(),
                    output_type,
                    ldd,
                    compute_type);
    }

    /**
     * Helper method to create that subset of a long hipblaslt argument list that is common
     * to multiple "gemm_ex..." calls.
     *
     * The hipblaslt GEMM API handles inputs and output matrices as
     *  column-major format. When doing a C = A * B, we actually do
     *   C^T = (B^T) * (A^T). That is the reason we input args[1] as
     *   A and args[0] as B in calling the hipblaslt GEMM.
     *
     * */
    auto create_hipblaslt_args_common(context& ctx, const std::vector<argument>& args, int32_t solution_idx)
    {
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha(),     // alpha
                    args[1].data(),  // A
                    matA,            // Adesc
                    args[0].data(),  // B
                    matB,            // Bdesc
                    get_beta(),      // beta
                    args[2].data(),  // C
                    matC,            // Cdesc
                    is_3inputs ? args[3].data() : args[2].data(),  // D
                    is_3inputs ? matD: matC,                       // Ddesc
                    &solution.get_result(ctx, *this, solution_idx)[0].algo,                     // algo
                    ctx.get_stream().get_hipblaslt_workspace(),    // workspace
                    HIPBLASLT_WORKSPACE_SIZE,                      // workspaceSizeInBytes
                    ctx.get_stream().get()                         // stream
                    );
    }
#ifdef MIGRAPHX_USE_ROCBLAS_TUNING_API
    auto create_hipblaslt_tuning_args_common(context& ctx, const std::vector<argument>& args, std::vector<hipblasLtMatmulHeuristicResult_t> result) const
    {
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                    op_A,
                    op_B,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    HIPBLASLT_COMPUTE_F32,
                    result
                    );
    }

    auto create_hipblaslt_supporting_args_common(context& ctx, const std::vector<argument>& args, hipblasLtMatmulAlgo_t &algo, size_t &workspace_size) const
    {
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha(),
                    matA,
                    matB,
                    get_beta(),
                    matC,
                    matC,
                    algo,
                    workspace_size
                    );
    }
#endif

#ifdef MIGRAPHX_USE_ROCBLAS_TUNING_API
    /**
     * Find best rocBLAS solution:  Get list of solutions and try them all, returning the index
     * of the fastest one.
     */
    int tune(context& ctx, const std::vector<shape>& input_shapes) //const
    {
        // tuning meta parameters
        const int hot_calls = 40;

        // First part: rocBLAS
        std::vector<argument> input_args;
        std::transform(input_shapes.begin(),
                       input_shapes.end(),
                       std::back_inserter(input_args),
                       [](const shape& x) { return to_gpu(generate_argument(x)); });

        // Get the solutions list in 2 rocBLAS steps:
        // 1.  Find out how many solutions there are and allocate the array
        // 2.  Get the solutions
        //
        rocblas_int list_size = 0;
        std::vector<rocblas_int> solution_indices_0;
        if(strided_batched)
        {
            auto common_args = create_strided_batched_args_common(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex_get_solutions,
                           common_args,
                           rocblas_gemm_algo_solution_index,
                           gemm_flags,
                           nullptr,
                           &list_size);
            solution_indices_0.resize(list_size);

            auto common_sol_args = create_strided_batched_args_common(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_strided_batched_ex_get_solutions,
                           common_sol_args,
                           rocblas_gemm_algo_solution_index,
                           gemm_flags,
                           solution_indices_0.data(),
                           &list_size);
        }
        else
        {
            auto common_args = create_gemm_ex_args_common(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_ex_get_solutions,
                           common_args,
                           rocblas_gemm_algo_solution_index,
                           gemm_flags,
                           nullptr,
                           &list_size);
            solution_indices_0.resize(list_size);

            auto common_sol_args = create_gemm_ex_args_common(ctx, input_args);
            rocblas_invoke(&rocblas_gemm_ex_get_solutions,
                           common_sol_args,
                           rocblas_gemm_algo_solution_index,
                           gemm_flags,
                           solution_indices_0.data(),
                           &list_size);
        }

        // Second part: hipblasLt
        std::vector<hipblasLtMatmulHeuristicResult_t> result;
#if 1
        hipblaslt_ext::getAllAlgos(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                    op_A,
                    op_B,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    HIPBLASLT_COMPUTE_F32,
                    result
                    );
#else
        auto tuning_args = create_hipblaslt_tuning_args_common(ctx, input_args, result);
        hipblaslt_invoke(&hipblaslt_ext::getAllAlgos,
                         tuning_args
                         );

#endif
        std::vector<int> solution_indices_1;
        int returned_algo_count = result.size();
        for (int i = 0; i < returned_algo_count; i++)
        {
            auto algo = result[i].algo;
            size_t ret_workspace_size = 0;
   
#if 1
            auto status = hipblaslt_ext::matmulIsAlgoSupported(
                                ctx.get_stream().get_hipblaslt(),
                                hipblaslt_desc,
                                get_alpha(),
                                matA,
                                matB,
                                get_beta(),
                                matC,
                                matC,
                                algo,
                                ret_workspace_size
                                );
#else
            auto supporting_args = create_hipblaslt_supporting_args_common(ctx, input_args, algo, ret_workspace_size);
            auto status = hipblaslt_invoke(&hipblaslt_ext::matmulIsAlgoSupported,
                            supporting_args
                            );
#endif
            if (status == HIPBLAS_STATUS_SUCCESS)
            { 
                if (ret_workspace_size < HIPBLASLT_WORKSPACE_SIZE)
                {
                    solution_indices_1.push_back(hipblaslt_ext::getIndexFromAlgo(algo));
                }
                else
                {
                    std::cout << "Need space larger than given workspace!" << std::endl;
                }
            }
        }

        // Third part: comparing rocBLAS and hipblasLt solutions
        double best_time  = std::numeric_limits<double>::max();
        double first_time = -1;
        std::vector<int32_t> result_0, result_1;
        std::vector<int32_t> solution_indices;

        std::transform(solution_indices_0.begin(), solution_indices_0.end(), std::back_inserter(result_0),
                       [this](int32_t elem) { return this->AS_MGX_IDX(elem, false); });
        std::transform(solution_indices_1.begin(), solution_indices_1.end(), std::back_inserter(result_1),
                       [this](int32_t elem) { return this->AS_MGX_IDX(elem, true); });
        std::copy(result_0.begin(), result_0.end(), std::back_inserter(solution_indices));
        std::copy(result_1.begin(), result_1.end(), std::back_inserter(solution_indices));

        // Initialize to default solution index
        rocblas_int best_sol = 0;
        for(auto sol : solution_indices)
        {
            // Warmup: the first call to an op. may not be representative since there is
            // more time taken initializing caches, etc. so we won't time it.
            run(ctx, input_args, sol);
            double host_time = time<milliseconds>([&] {
                for([[maybe_unused]] int hc : range(hot_calls))
                    run(ctx, input_args, sol);
                ctx.finish();
            });

            host_time /= hot_calls;

            // dev/evaluation only: track time for first solution.
            if(first_time < 0)
                first_time = host_time;

            // track current best
            if(host_time < best_time)
            {
                best_sol  = sol;
                best_time = host_time;
            }
        }

        std::cout << "Winning GEMM solution: "
                  << (AS_BLAS_IDX(best_sol))
                  << " (" << (IS_ROCBLAS(best_sol) ? "rocblas" : "hipblaslt") << ")"
                  << " in " << best_time << " ms, beats "
                  << first_time << "ms" << std::endl;
        return best_sol;
    }
#endif
    private:
    // rocblas
    size_t num_matrices = 0;
    rocblas_int m       = 0;
    rocblas_int n       = 0;
    rocblas_int k       = 0;
    bool transa         = false;
    bool transb         = false;
    T alpha             = 0;
    T beta              = 0;

    std::function<const void*()> get_alpha{};
    std::function<const void*()> get_beta{};
    rocblas_gemm_flags gemm_flags = rocblas_gemm_flags_none;
    rocblas_int lda               = 0;
    rocblas_int ldb               = 0;
    rocblas_int ldc               = 0;
    rocblas_int ldd               = 0;
    rocblas_int a_stride          = 0;
    rocblas_int b_stride          = 0;
    rocblas_int c_stride          = 0;
    rocblas_int d_stride          = 0;
    rocblas_datatype arg_type     = rocblas_datatype_f32_r;
    rb_compute_type compute_type  = rocblas_datatype_f32_r;
    rocblas_datatype output_type  = rocblas_datatype_f32_r;
    bool strided_batched          = true;
    bool is_3inputs               = true;
    bool compute_fp32             = true;

    // hipblaslt
    hipblasltDatatype_t dtype;
    hipblasLtMatmulDesc_t hipblaslt_desc;
    hipblasOperation_t op_A;
    hipblasOperation_t op_B;
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtHandle_t handle;
    hipblasLtMatmulPreference_t preference;
}; // gemm_impl

void gemm_compute(context& ctx,
                  const shape& output_shape,
                  const std::vector<argument>& args,
                  float alpha,
                  float beta,
                  bool compute_fp32,
                  int32_t solution_idx)
{
    std::vector<shape> input_shapes;
    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(input_shapes),
                   [](const argument& x) { return x.get_shape(); });
    auto gemm_item = gemm_impl<float>(output_shape, input_shapes, alpha, beta, compute_fp32);
    gemm_item.run(ctx, args, solution_idx);
}

void gemm_compute(context& ctx,
                  const shape& output_shape,
                  const std::vector<argument>& args,
                  int32_t alpha,
                  int32_t beta,
                  bool compute_fp32,
                  int32_t solution_idx)
{
    std::vector<shape> input_shapes;
    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(input_shapes),
                   [](const argument& x) { return x.get_shape(); });
    auto gemm_item = gemm_impl<int32_t>(output_shape, input_shapes, alpha, beta, compute_fp32);

    gemm_item.run(ctx, args, solution_idx);
}

static value gemm_problem(const shape& output_shape, std::vector<shape> input_shapes)
{
    input_shapes.push_back(output_shape);
    return to_value(input_shapes);
}

#ifdef MIGRAPHX_USE_ROCBLAS_TUNING_API
static void gemm_save_solution(context& ctx,
                               const shape& output_shape,
                               const std::vector<shape>& input_shapes,
                               int32_t solution_idx)
{
    ctx.get_problem_cache().insert(
        "rocblas", gemm_problem(output_shape, input_shapes), solution_idx);
}
#endif

int32_t gemm_default_solution(context& ctx,
                              const shape& output_shape,
                              const std::vector<shape>& input_shapes)
{
    auto sol = ctx.get_problem_cache().get("rocblas", gemm_problem(output_shape, input_shapes));
    if(sol.has_value())
        return sol->to<int32_t>();
    return 0;
}

/**
 * Decides if the tune() or validate() method is appropriate and calls it.
 * Return value is the chosen solution index, or 0 to let picker choose it.
 */
template <class T>
int32_t gemm_finalize_impl(context& ctx,
                           const shape& output_shape,
                           const std::vector<shape>& input_shapes,
                           T alpha,
                           T beta,
                           bool compute_fp32,
                           int32_t solution_idx)
{
#ifdef MIGRAPHX_USE_ROCBLAS_TUNING_API

    // This code should be called only if either the environment var.
    // MIGRAPHX_ENABLE_GEMM_TUNING, or option --exhaustive-tune, is set

    if(solution_idx == 0)
    {
        auto gemm_item = gemm_impl<T>(output_shape, input_shapes, alpha, beta, compute_fp32);
        solution_idx   = gemm_item.tune(ctx, input_shapes);
        gemm_save_solution(ctx, output_shape, input_shapes, solution_idx);
    }
    else
    {
        // If a tuned solution index is already given, don't tune again but validate
        // in case the data was tuned with a different rocBLAS version
        auto gemm_item = gemm_impl<T>(output_shape, input_shapes, alpha, beta, compute_fp32);
        solution_idx   = gemm_item.validate(ctx, input_shapes, solution_idx);
    }
#else
    (void)ctx, (void)output_shape, (void)input_shapes;
    (void)alpha, (void)beta, (void)compute_fp32;
#endif
    return solution_idx;
}

int32_t gemm_finalize(context& ctx,
                      const shape& output_shape,
                      const std::vector<shape>& input_shapes,
                      float alpha,
                      float beta,
                      bool compute_fp32,
                      int32_t solution_idx)
{
    return gemm_finalize_impl(
        ctx, output_shape, input_shapes, alpha, beta, compute_fp32, solution_idx);
}

int32_t gemm_finalize(context& ctx,
                      const shape& output_shape,
                      const std::vector<shape>& input_shapes,
                      int32_t alpha,
                      int32_t beta,
                      bool compute_fp32,
                      int32_t solution_idx)
{
    return gemm_finalize_impl(
        ctx, output_shape, input_shapes, alpha, beta, compute_fp32, solution_idx);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
