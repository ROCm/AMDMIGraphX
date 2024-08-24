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
#include <limits>
#include <migraphx/gpu/hipblaslt.hpp>
#include <migraphx/gpu/hip_gemm_impl.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/time.hpp>

using microseconds = std::chrono::duration<double, std::micro>;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

hipDataType compute_to_hip_type(hipblasComputeType_t type)
{
    switch(type)
    {
    case HIPBLAS_COMPUTE_32F: return HIP_R_32F;
    case HIPBLAS_COMPUTE_32I: return HIP_R_32I;
    case HIPBLAS_COMPUTE_16F:
    case HIPBLAS_COMPUTE_64F:
    case HIPBLAS_COMPUTE_32I_PEDANTIC:
    case HIPBLAS_COMPUTE_16F_PEDANTIC:
    case HIPBLAS_COMPUTE_32F_PEDANTIC:
    case HIPBLAS_COMPUTE_64F_PEDANTIC:
    case HIPBLAS_COMPUTE_32F_FAST_16F:
    case HIPBLAS_COMPUTE_32F_FAST_16BF:
    case HIPBLAS_COMPUTE_32F_FAST_TF32:
        MIGRAPHX_THROW("HIPBLAS_GEMM: conversion from hipComputeType_t to hipDataType failed");
    }
}

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
 * Wrapper for multiple hipBLASLt calls.  The constructor creates parameters for
 * these calls based on data shapes and other values contained in the associated
 * instruction and operation.
 */
struct hip_gemm_impl
{
    hip_gemm_impl(const shape& output_shape,
                  const std::vector<shape>& input_shapes,
                  float alpha_param,
                  float beta_param)
        : solution(), alpha(alpha_param), beta(beta_param), is_3inputs(input_shapes.size() == 4)
    {
        if(not is_3inputs)
        {
            beta = 0;
        }

        // Create lambdas that will cast alpha, beta to the output shape's type
        // and retain the values being pointed to
        output_shape.visit_type([&](auto as) {
            if(as.is_integral())
            {
                int32_t alpha_r = int32_t(alpha);
                int32_t beta_r  = int32_t(beta);
                get_alpha       = [=] { return &alpha_r; };
                get_beta        = [=] { return &beta_r; };
            }
            else
            {
                get_alpha = [=] { return &alpha; };
                get_beta  = [=] { return &beta; };
            }
        });

        transa = is_transposed_hip(input_shapes[0]);
        transb = is_transposed_hip(input_shapes[1]);
        op_A   = transa ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        op_B   = transb ? HIPBLAS_OP_T : HIPBLAS_OP_N;

        auto n_dim = output_shape.lens().size();
        auto dim_0 = n_dim - 2;
        auto dim_1 = n_dim - 1;
        // Leading dimensions of matrices
        lda = input_shapes[0].strides()[transa ? dim_1 : dim_0];
        ldb = input_shapes[1].strides()[transb ? dim_1 : dim_0];
        ldc = input_shapes[2].strides()[dim_0];
        ldd = is_3inputs ? input_shapes[3].strides()[dim_0] : ldc;

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

        arg_type    = get_type_hipblas(input_shapes[0].type());
        output_type = get_type_hipblas(input_shapes[2].type());

        if(arg_type == HIP_R_8I or arg_type == HIP_R_8U)
        {
            compute_type = HIPBLAS_COMPUTE_32I;
        }
        else
        {
            compute_type = HIPBLAS_COMPUTE_32F;
        }
        if(op_A == HIPBLAS_OP_T)
        {
            hipblaslt_invoke(
                [&]() { return hipblasLtMatrixLayoutCreate(&matA, arg_type, m, k, lda); });
        }
        else
        {
            hipblaslt_invoke(
                [&]() { return hipblasLtMatrixLayoutCreate(&matA, arg_type, k, m, lda); });
        }
        if(op_B == HIPBLAS_OP_T)
        {
            hipblaslt_invoke(
                [&]() { return hipblasLtMatrixLayoutCreate(&matB, arg_type, k, n, ldb); });
        }
        else
        {
            hipblaslt_invoke(
                [&]() { return hipblasLtMatrixLayoutCreate(&matB, arg_type, n, k, ldb); });
        }
        hipblaslt_invoke(
            [&]() { return hipblasLtMatrixLayoutCreate(&matC, output_type, n, m, ldc); });

        if(is_3inputs)
        {
            hipblaslt_invoke(
                [&]() { return hipblasLtMatrixLayoutCreate(&matD, output_type, n, m, ldd); });
        }
        if(num_matrices > 1)
        {
            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matA, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices));
            });
            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matB, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices));
            });
            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matC, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &num_matrices, sizeof(num_matrices));
            });

            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matA,
                    HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &a_stride,
                    sizeof(a_stride));
            });
            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matB,
                    HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &b_stride,
                    sizeof(b_stride));
            });
            hipblaslt_invoke([&]() {
                return hipblasLtMatrixLayoutSetAttribute(
                    matC,
                    HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &c_stride,
                    sizeof(c_stride));
            });

            if(is_3inputs)
            {
                hipblaslt_invoke([&]() {
                    return hipblasLtMatrixLayoutSetAttribute(matD,
                                                             HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                             &num_matrices,
                                                             sizeof(num_matrices));
                });
                hipblaslt_invoke([&]() {
                    return hipblasLtMatrixLayoutSetAttribute(
                        matD,
                        HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                        &d_stride,
                        sizeof(d_stride));
                });
            }
        }
        hipblaslt_invoke([&]() {
            return hipblasLtMatmulDescCreate(
                &hipblaslt_desc, compute_type, compute_to_hip_type(compute_type));
        });
        hipblaslt_invoke([&]() {
            return hipblasLtMatmulDescSetAttribute(
                hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_A, sizeof(int32_t));
        });
        hipblaslt_invoke([&]() {
            return hipblasLtMatmulDescSetAttribute(
                hipblaslt_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_B, sizeof(int32_t));
        });

        // Transfer ownership of raw pointers to managed pointers.
        managed_hipblaslt_desc.reset(hipblaslt_desc);
        managed_matA.reset(matA);
        managed_matB.reset(matB);
        managed_matC.reset(matC);
        if(is_3inputs)
        {
            managed_matD.reset(matD);
        }
    }

    ~hip_gemm_impl() {}

    struct solution
    {
        solution() : handle(nullptr), preference(nullptr) {}

        auto get_hipblaslt_preference()
        {
            if(hbltpreference == nullptr)
            {
                hbltpreference = create_hipblaslt_preference_ptr();
            }
            assert(hbltpreference.get() != nullptr);
            return hbltpreference.get();
        }

        void init(context& ctx)
        {
            if(handle == nullptr)
            {
                handle     = ctx.get_stream().get_hipblaslt();
                preference = get_hipblaslt_preference();
            }
        }

        auto& get_result(context& ctx, hip_gemm_impl& gemm, int32_t idx)
        {
            init(ctx);
            if(idx == 0)
            {
                // use default solution
                const int n_sol = 1;
                int returnedAlgoCount;
                heuristicResult.resize(n_sol);
                uint64_t max_workspace = std::numeric_limits<uint64_t>::max();
                hipblaslt_invoke([&]() {
                    return hipblasLtMatmulPreferenceSetAttribute(
                        preference,
                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &max_workspace,
                        sizeof(uint64_t));
                });
                hipblaslt_invoke([&]() {
                    return hipblasLtMatmulAlgoGetHeuristic(handle,
                                                           gemm.hipblaslt_desc,
                                                           gemm.matB,
                                                           gemm.matA,
                                                           gemm.matC,
                                                           gemm.is_3inputs ? gemm.matD : gemm.matC,
                                                           preference,
                                                           n_sol,
                                                           heuristicResult.data(),
                                                           &returnedAlgoCount);
                });

                if(returnedAlgoCount != n_sol)
                {
                    std::cout << "less solution found! request: " << n_sol
                              << ", found: " << returnedAlgoCount << std::endl;
                }
            }
            else
            {
                // query for the solutions. 1st as the best.
                std::vector<int32_t> algoIndex = {idx};
                hipblaslt_invoke([&]() {
                    return hipblaslt_ext::getAlgosFromIndex(handle, algoIndex, heuristicResult);
                });
                assert(heuristicResult.size() == 1);
            }
            return heuristicResult;
        }

        private:
        hipblasLtHandle_t handle;
        hipblasLtMatmulPreference_t preference;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        shared<hipblaslt_preference_ptr> hbltpreference = nullptr;
    } solution;

    /**
     * Helper method to create that subset of a long hipblaslt argument list that is common
     * to multiple "hipblasLtMatmul" calls.
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
        auto algo = &solution.get_result(ctx, *this, solution_idx)[0].algo;
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha(),                                  // alpha
                    args[1].data(),                               // A
                    matB,                                         // Adesc
                    args[0].data(),                               // B
                    matA,                                         // Bdesc
                    get_beta(),                                   // beta
                    args[2].data(),                               // C
                    matC,                                         // Cdesc
                    is_3inputs ? args[3].data() : args[2].data(), // D
                    is_3inputs ? matD : matC,                     // Ddesc
                    algo,                                         // algo
                    ctx.get_stream().get_hipblaslt_workspace(),   // workspace
                    algo->max_workspace_bytes,                    // workspaceSizeInBytes
                    ctx.get_stream().get()                        // stream
        );
    }

    auto create_hipblaslt_supporting_args_common(context& ctx,
                                                 const std::vector<argument>& args,
                                                 hipblasLtMatmulAlgo_t& algo,
                                                 size_t& workspace_size) const
    {
        (void)(args);
        return pack(ctx.get_stream().get_hipblaslt(),
                    hipblaslt_desc,
                    get_alpha(),
                    matB,
                    matA,
                    get_beta(),
                    matC,
                    is_3inputs ? matD : matC,
                    algo,
                    workspace_size);
    }

    void
    run(context& ctx, const std::vector<argument>& input_args, int32_t solution_idx = 0) // const
    {
        auto common_args = create_hipblaslt_args_common(ctx, input_args, solution_idx);
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
     * Checks a particular solution for validity by running it (could be invalid if this model was
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
     * Find best hipBLASLt solution:  Get list of solutions and try them all, returning the index
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

        std::vector<hipblasLtMatmulHeuristicResult_t> result;
        hipblaslt_invoke([&]() {
            return hipblaslt_ext::getAllAlgos(ctx.get_stream().get_hipblaslt(),
                                              hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                              op_A,
                                              op_B,
                                              arg_type,
                                              arg_type,
                                              output_type,
                                              output_type,
                                              compute_type,
                                              result);
        });
        std::vector<int32_t> solution_indices;
        int returned_algo_count = result.size();
        for(int i = 0; i < returned_algo_count; i++)
        {
            auto algo                 = result[i].algo;
            size_t ret_workspace_size = 0;
            auto supporting_args =
                create_hipblaslt_supporting_args_common(ctx, input_args, algo, ret_workspace_size);
            try
            {
                hipblaslt_invoke(&hipblaslt_ext::matmulIsAlgoSupported, supporting_args);
                solution_indices.push_back(hipblaslt_ext::getIndexFromAlgo(algo));
            }
            catch(...)
            {
                // algo is not supported, continue in that case
                continue;
            }
        }

        double best_time  = std::numeric_limits<double>::max();
        double first_time = -1;

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
    uint64_t m          = 0;
    uint64_t n          = 0;
    uint64_t k          = 0;
    bool transa         = false;
    bool transb         = false;
    float alpha         = 0;
    float beta          = 0;
    std::function<const void*()> get_alpha{};
    std::function<const void*()> get_beta{};

    int64_t lda      = 0;
    int64_t ldb      = 0;
    int64_t ldc      = 0;
    int64_t ldd      = 0;
    int64_t a_stride = 0;
    int64_t b_stride = 0;
    int64_t c_stride = 0;
    int64_t d_stride = 0;
    bool is_3inputs  = true;

    // hipblaslt
    hipDataType arg_type              = HIP_R_32F;
    hipblasComputeType_t compute_type = HIPBLAS_COMPUTE_32F;
    hipDataType output_type           = HIP_R_32F;
    hipblasLtMatmulDesc_t hipblaslt_desc;
    hipblasOperation_t op_A;
    hipblasOperation_t op_B;
    using hipblaslt_matrix_layout = MIGRAPHX_MANAGE_PTR(hipblasLtMatrixLayout_t,
                                                        hipblasLtMatrixLayoutDestroy);
    using hipblaslt_mat_mul_desc  = MIGRAPHX_MANAGE_PTR(hipblasLtMatmulDesc_t,
                                                       hipblasLtMatmulDescDestroy);
    hipblaslt_matrix_layout managed_matA, managed_matB, managed_matC, managed_matD;
    hipblaslt_mat_mul_desc managed_hipblaslt_desc;
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    hipblasLtHandle_t handle;
    hipblasLtMatmulPreference_t preference;
}; // hip_gemm_impl

void hip_gemm_compute(context& ctx,
                      const shape& output_shape,
                      const std::vector<argument>& args,
                      float alpha,
                      float beta,
                      int32_t solution_idx)
{
    std::vector<shape> input_shapes;
    std::transform(args.begin(),
                   args.end(),
                   std::back_inserter(input_shapes),
                   [](const argument& x) { return x.get_shape(); });
    auto gemm_item = hip_gemm_impl(output_shape, input_shapes, alpha, beta);
    gemm_item.run(ctx, args, solution_idx);
}

static value hip_gemm_problem(const shape& output_shape, std::vector<shape> input_shapes)
{
    input_shapes.push_back(output_shape);
    return to_value(input_shapes);
}

static void hip_gemm_save_solution(context& ctx,
                                   const shape& output_shape,
                                   const std::vector<shape>& input_shapes,
                                   int32_t solution_idx)
{
    ctx.get_problem_cache().insert(
        "hipblaslt", hip_gemm_problem(output_shape, input_shapes), solution_idx);
}

int32_t hip_gemm_finalize(context& ctx,
                          const shape& output_shape,
                          const std::vector<shape>& input_shapes,
                          float alpha,
                          float beta,
                          int32_t solution_idx)
{
    auto gemm_item   = hip_gemm_impl(output_shape, input_shapes, alpha, beta);
    int32_t solution = gemm_item.tune(ctx, input_shapes);
    hip_gemm_save_solution(ctx, output_shape, input_shapes, solution_idx);
    return solution;
}

int32_t hip_gemm_default_solution(context& ctx,
                                  const shape& output_shape,
                                  const std::vector<shape>& input_shapes)
{
    auto sol =
        ctx.get_problem_cache().get("hipblaslt", hip_gemm_problem(output_shape, input_shapes));
    if(sol.has_value())
        return sol->to<int32_t>();
    return 0;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
