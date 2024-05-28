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

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <migraphx/gpu/hipblaslt.hpp>
#include <migraphx/gpu/hip_gemm_impl.hpp>
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

// Convert hipBLAS datatypes to equivalent Migraphx data types
hipDataType get_type_hipblas(shape::type_t type)
{
    switch(type)
    {
    case shape::double_type: return HIP_R_64F;
    case shape::float_type: return HIP_R_32F;
    case shape::half_type: return HIP_R_16F;
    case shape::int8_type: return HIP_R_8I;
    case shape::uint8_type: return HIP_R_8U;
    case shape::int32_type: return HIP_R_32I;
    case shape::uint32_type: return HIP_R_32U;
    case shape::fp8e4m3fnuz_type: return HIP_R_8F_E4M3_FNUZ;
    case shape::tuple_type:
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type: MIGRAPHX_THROW("HIPBLAS_GEMM: data type not supported!");
    }

    MIGRAPHX_THROW("HIPBLAS_GEMM: data type not supported!");
}

template <class F, class Pack, class... Ts>
auto hipblaslt_invoke(F f, Pack p, Ts... xs)
{
    return p([=](auto... ws) {
        auto status = f(ws..., xs...);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            MIGRAPHX_THROW("hipblaslt_invoke: hipBlasLt call failed with status " +
                           std::to_string(status));
        }
        return status;
    });
}

void blas_shape_hip(const shape& s)
{
    if(s.lens().size() < 2)
        return;
    if(std::none_of(s.strides().end() - 2, s.strides().end(), [](auto i) { return i == 1; }))
        MIGRAPHX_THROW("GPU_GEMM: needs to have one matrix stride as 1");
    if(std::any_of(s.strides().end() - 2, s.strides().end(), [](auto i) { return i == 0; }))
        MIGRAPHX_THROW("GPU_GEMM: matrix dimensions can't be broadcasted");
    if(s.lens().size() < 3)
        return;
    shape batch_shape{s.type(),
                      {s.lens().begin(), s.lens().end() - 2},
                      {s.strides().begin(), s.strides().end() - 2}};
    auto batch_shapes = reduce_dims({batch_shape});
    if(batch_shapes.front().lens().size() != 1)
        MIGRAPHX_THROW("GPU_GEMM: Batch dimension is not collapsible");
}

shape transpose_batch_hip(const shape& s, unsigned trans_batch)
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
static bool is_transposed_hip(const shape& s) { return s.transposed() and s.strides().back() != 1; }

static int32_t get_batch_stride_hip(const shape& s)
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
struct hip_gemm_impl
{
    hip_gemm_impl(const shape& output_shape,
              const std::vector<shape>& input_shapes,
              T alpha_param,
              T beta_param,
              bool compute_fp32_flag)
        : solution(),alpha(alpha_param),
          beta(beta_param),
          is_3inputs(input_shapes.size() == 4),
          compute_fp32(compute_fp32_flag)
    {
        std::cout << "inside constructor \n";
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
                get_alpha_hip = [=] { return &alpha; };
                get_beta_hip  = [=] { return &beta; };
            }
            else
            {
                get_alpha_hip = [=] { return &alpha_r; };
                get_beta_hip  = [=] { return &beta_r; };
            }
        });

        transa     = is_transposed_hip(input_shapes[0]);
        transb     = is_transposed_hip(input_shapes[1]);
        auto n_dim = output_shape.lens().size();
        auto dim_0 = n_dim - 2;
        auto dim_1 = n_dim - 1;
        // Leading dimensions of matrices
        lda = input_shapes[0].strides()[transa ? dim_1 : dim_0];
        ldb = input_shapes[1].strides()[transb ? dim_1 : dim_0];
        ldc = input_shapes[2].strides()[dim_0];
        ldd = is_3inputs ? input_shapes[3].strides()[dim_0] : ldc;

        arg_type    = get_type_hipblas(input_shapes[0].type());
        output_type = get_type_hipblas(input_shapes[2].type());
        if(output_type == HIP_R_8I or output_type == HIP_R_8U)
        {
            output_type = HIP_R_32I;
        }
        compute_type = HIP_R_32F;
        if(compute_fp32)
        {
            if(arg_type == HIP_R_16F)
                compute_type = HIP_R_32F;
        }
        if(arg_type == HIP_R_8F_E4M3_FNUZ)
        {
            assert(get_type_hipblas(input_shapes[1].type()) == HIP_R_8F_E4M3_FNUZ);
            compute_type = HIP_R_32F;
        }

        auto a_lens = input_shapes[0].lens();
        auto b_lens = input_shapes[1].lens();

        auto out_lens = output_shape.lens();
        m             = out_lens[dim_0];
        n             = out_lens[dim_1];
        k             = input_shapes[0].lens()[dim_1];

        a_stride     = get_batch_stride_hip(input_shapes[0]);
        b_stride     = get_batch_stride_hip(input_shapes[1]);
        c_stride     = get_batch_stride_hip(input_shapes[2]);
        d_stride     = is_3inputs ? get_batch_stride_hip(input_shapes[3]) : c_stride;
        num_matrices = std::accumulate(
            out_lens.rbegin() + 2, out_lens.rend(), std::size_t{1}, std::multiplies<std::size_t>());
        strided_batched       = num_matrices > 1;
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

        // casting from int32_t which is int64_t
        const uint64_t m_ = static_cast<uint64_t>(n);
        const uint64_t n_ = static_cast<uint64_t>(rocblas_fallback ? m / num_matrices : m);
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
        std::cout << "inside constructor\n";
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
                CHECK_HIPBLAS_ERROR(
                    hipblasLtMatrixLayoutSetAttribute(matD,
                                                      HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                      &num_matrices,
                                                      sizeof(num_matrices)));
                CHECK_HIPBLAS_ERROR(
                    hipblasLtMatrixLayoutSetAttribute(matD,
                                                      HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                                      &d_stride_,
                                                      sizeof(d_stride_)));
            }
        }
        if(hipblaslt_desc == nullptr)
        {
            std::cout << "Just before setting,  it isnullptr\n";
        }
        CHECK_HIPBLAS_ERROR(
            hipblasLtMatmulDescCreate(&hipblaslt_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(int32_t)));
        CHECK_HIPBLAS_ERROR(hipblasLtMatmulDescSetAttribute(
            hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(int32_t)));
        if(hipblaslt_desc == nullptr)
        {
            std::cout << "init of matmul desc failed\n";
        }
        else
        {
            std::cout << "init successful inside constructor\n";
        }
        std::cout << "leaving constructor\n";
    }

    ~hip_gemm_impl()
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
            if(handle == nullptr)
            {
                std::cout << "handle is nullptr\n";
                handle     = ctx.get_stream().get_hipblaslt();
                preference = ctx.get_stream().get_hipblaslt_preference();
            }
        }

        auto& get_result(context& ctx, hip_gemm_impl& gemm, int32_t idx)
        {
            init(ctx);
            std::cout << "done initialization\n";
            if(idx == 0)
            {
                std::cout << "solution index is zero\n";
                // use default solution
                const int n_sol = 1;
                int returnedAlgoCount;
                heuristicResult.resize(n_sol);
                if(handle == nullptr)
                {
                    std::cout << "not initialized\n";
                }
                if(gemm.matA == nullptr or gemm.matB == nullptr or gemm.matC == nullptr)
                {
                    std::cout << "gemm is null\n";
                }
                if(gemm.hipblaslt_desc == nullptr)
                {
                    std::cout << "desc is null\n";
                }

                if(preference == nullptr)
                {
                    std::cout << "Preference is nullptr\n";
                }
                auto status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                              gemm.hipblaslt_desc,
                                                              gemm.matA,
                                                              gemm.matB,
                                                              gemm.matC,
                                                              gemm.matC,
                                                              preference,
                                                              n_sol,
                                                              heuristicResult.data(),
                                                              &returnedAlgoCount);
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    std::cout << "algo failed\n";
                    std::abort();
                }
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
                CHECK_HIPBLAS_ERROR(
                    hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResult));
            }
            return heuristicResult;
        }

        private:
        hipblasLtHandle_t handle;
        hipblasLtMatmulPreference_t preference;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    } solution;

    void
    run(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx = 0) // const
    {
        auto common_args = create_hipblaslt_args_common(ctx, input_args, solution_idx);
        std::cout << "Done creating arguments\n";
        hipblaslt_invoke(&hipblasLtMatmul, common_args);
    }

    auto
    validate(context& ctx, const std::vector<shape>& input_shapes, int32_t solution_idx) // const
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
    validate(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx) // const
    {
        hipblasStatus_t check_valid(HIPBLAS_STATUS_SUCCESS);
        auto common_args = create_hipblaslt_args_common(ctx, input_args, solution_idx);
        check_valid      = hipblaslt_invoke(&hipblasLtMatmul, common_args);
        if(check_valid == HIPBLAS_STATUS_SUCCESS)
        {
            std::cerr << "WARNING:  tuned solution is invalid; reverting to default" << std::endl;
            return 0;
        }
        return solution_idx;
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
    auto create_hipblaslt_args_common(context& ctx,
                                      const std::vector<argument>& args,
                                      int32_t solution_idx)
    {
        std::cout << "creating args\n";
        std::cout << "args.size(): " << args.size() << std::endl;
        std::cout << "algo: " << &solution.get_result(ctx, *this, solution_idx)[0].algo
                  << std::endl;
        std::cout << "alpha: " << get_alpha_hip() << std::endl;
        std::cout << "beta: " << get_beta_hip() << std::endl;
        auto algo = &solution.get_result(ctx, *this, solution_idx)[0].algo;
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha_hip(),                              // alpha
                    args[1].data(),                               // A
                    matA,                                         // Adesc
                    args[0].data(),                               // B
                    matB,                                         // Bdesc
                    get_beta_hip(),                               // beta
                    args[2].data(),                               // C
                    matC,                                         // Cdesc
                    is_3inputs ? args[3].data() : args[2].data(), // D
                    is_3inputs ? matD : matC,                     // Ddesc
                    algo,                                         // algo
                    ctx.get_stream().get_hipblaslt_workspace(),   // workspace
                    HIPBLASLT_WORKSPACE_SIZE,                     // workspaceSizeInBytes
                    ctx.get_stream().get()                        // stream
        );
    }

    auto
    create_hipblaslt_tuning_args_common(context& ctx,
                                        const std::vector<argument>& args,
                                        std::vector<hipblasLtMatmulHeuristicResult_t> result) const
    {
        (void)(args);
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                    op_A,
                    op_B,
                    dtype,
                    dtype,
                    dtype,
                    dtype,
                    HIPBLAS_COMPUTE_32F,
                    result);
    }

    auto create_hipblaslt_supporting_args_common(context& ctx,
                                                 const std::vector<argument>& args,
                                                 hipblasLtMatmulAlgo_t& algo,
                                                 size_t& workspace_size) const
    {
        (void)(args);
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha_hip(),
                    matA,
                    matB,
                    get_beta_hip(),
                    matC,
                    matC,
                    algo,
                    workspace_size);
    }

    /**
     * Find best rocBLAS solution:  Get list of solutions and try them all, returning the index
     * of the fastest one.
     */
    int tune(context& ctx, const std::vector<shape>& input_shapes) // const
    {
        // tuning meta parameters
        const int hot_calls = 40;

        std::vector<argument> input_args;
        std::transform(input_shapes.begin(),
                       input_shapes.end(),
                       std::back_inserter(input_args),
                       [](const shape& x) { return to_gpu(generate_argument(x)); });

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
                                   HIPBLAS_COMPUTE_32F,
                                   result);
#else
        auto tuning_args = create_hipblaslt_tuning_args_common(ctx, input_args, result);
        hipblaslt_invoke(&hipblaslt_ext::getAllAlgos, tuning_args);

#endif // if 1
        std::vector<int> solution_indices_1;
        int returned_algo_count = result.size();
        for(int i = 0; i < returned_algo_count; i++)
        {
            auto algo                 = result[i].algo;
            size_t ret_workspace_size = 0;

#if 1
            auto status = hipblaslt_ext::matmulIsAlgoSupported(ctx.get_stream().get_hipblaslt(),
                                                               hipblaslt_desc,
                                                               get_alpha_hip(),
                                                               matA,
                                                               matB,
                                                               get_beta_hip(),
                                                               matC,
                                                               matC,
                                                               algo,
                                                               ret_workspace_size);
#else
            auto supporting_args =
                create_hipblaslt_supporting_args_common(ctx, input_args, algo, ret_workspace_size);
            auto status = hipblaslt_invoke(&hipblaslt_ext::matmulIsAlgoSupported, supporting_args);
#endif // #if 1
            if(status == HIPBLAS_STATUS_SUCCESS)
            {
                if(ret_workspace_size < HIPBLASLT_WORKSPACE_SIZE)
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

        std::transform(solution_indices_1.begin(),
                       solution_indices_1.end(),
                       std::back_inserter(result_1),
                       [this](int32_t elem) { return elem; });
        std::copy(result_0.begin(), result_0.end(), std::back_inserter(solution_indices));
        std::copy(result_1.begin(), result_1.end(), std::back_inserter(solution_indices));

        // Initialize to default solution index
        int32_t best_sol = 0;
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

        std::cout << "Winning GEMM solution: " << best_sol << " in " << best_time << " ms, beats "
                  << first_time << "ms" << std::endl;
        return best_sol;
    }

    // rocblas
    size_t num_matrices = 0;
    int32_t m           = 0;
    int32_t n           = 0;
    int32_t k           = 0;
    bool transa         = false;
    bool transb         = false;
    T alpha             = 0;
    T beta              = 0;

    std::function<const void*()> get_alpha_hip{};
    std::function<const void*()> get_beta_hip{};
    int32_t lda              = 0;
    int32_t ldb              = 0;
    int32_t ldc              = 0;
    int32_t ldd              = 0;
    int32_t a_stride         = 0;
    int32_t b_stride         = 0;
    int32_t c_stride         = 0;
    int32_t d_stride         = 0;
    hipDataType arg_type     = HIP_R_32F;
    hipDataType compute_type = HIP_R_32F;
    hipDataType output_type  = HIP_R_32F;
    bool strided_batched     = true;
    bool is_3inputs          = true;
    bool compute_fp32        = true;

    // hipblaslt
    hipDataType dtype;
    hipblasLtMatmulDesc_t hipblaslt_desc;
    hipblasOperation_t op_A;
    hipblasOperation_t op_B;
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtHandle_t handle               = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
}; // hip_gemm_impl

void hip_gemm_compute(context& ctx,
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
    auto gemm_item = hip_gemm_impl<float>(output_shape, input_shapes, alpha, beta, compute_fp32);
    if(gemm_item.hipblaslt_desc == nullptr)
    {
        std::cout << "init failed\n";
    }
    gemm_item.run(ctx, args, solution_idx);
}

void hip_gemm_compute(context& ctx,
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
    auto gemm_item = hip_gemm_impl<int32_t>(output_shape, input_shapes, alpha, beta, compute_fp32);

    gemm_item.run(ctx, args, solution_idx);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
