/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/simple_par_for.hpp>
#include <migraphx/env.hpp>
#include <thread>
#include <unordered_set>

#if MIGRAPHX_USE_HIPBLASLT
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#endif

#if MIGRAPHX_USE_MIOPEN
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <migraphx/op/convolution.hpp>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_TRACE_PROPAGATE_CONSTANT)
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GPU_PROPAGATE)

static bool skip_propagate(instruction_ref ins)
{
    if(contains({"contiguous", "dequantizelinear", "reshape"}, ins->name()))
        return skip_propagate(ins->inputs().front());
    if(contains({"unpack_int4", "unpack_fp4"}, ins->name()))
        return true;
    auto&& s = ins->get_shape();
    if(s.broadcasted() and s.element_space() < s.elements())
        return true;
    auto alias = instruction::get_output_alias(ins, true);
    if(alias != ins)
        return skip_propagate(alias);
    if(ins->is_undefined())
        return true;
    return false;
}

static bool is_const_ins(instruction_ref ins, const std::unordered_set<std::string>& skip_ops)
{
    return ins->can_eval() and not skip_propagate(ins) and
           skip_ops.find(ins->name()) == skip_ops.end();
}

static argument as_packed(const argument& c)
{
    if(c.get_shape().packed())
        return c;
    auto s = c.get_shape().with_lens(c.get_shape().lens());
    argument result;
    c.visit([&](auto x) { result = literal{s, x.begin(), x.end()}.get_argument(); });
    return result;
}

#if MIGRAPHX_USE_HIPBLASLT

static hipDataType get_hipblaslt_type(shape::type_t type)
{
    switch(type)
    {
    case shape::double_type: return HIP_R_64F;
    case shape::float_type: return HIP_R_32F;
    case shape::half_type: return HIP_R_16F;
    case shape::bf16_type: return HIP_R_16BF;
    case shape::int8_type: return HIP_R_8I;
    case shape::uint8_type: return HIP_R_8U;
    case shape::int32_type: return HIP_R_32I;
    case shape::uint32_type: return HIP_R_32U;
    case shape::fp8e4m3fnuz_type: return HIP_R_8F_E4M3_FNUZ;
    case shape::fp8e5m2fnuz_type: return HIP_R_8F_E5M2_FNUZ;
    case shape::fp8e4m3fn_type: return HIP_R_8F_E4M3;
    case shape::fp8e5m2_type: return HIP_R_8F_E5M2;
    case shape::bool_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type:
    case shape::tuple_type:
    case shape::fp4x2_type: return HIP_R_32F;
    }
    return HIP_R_32F;
}

// Thread-safe cached hipBLASLt handle. Created once on first use.
static hipblasLtHandle_t get_hipblaslt_handle()
{
    static hipblasLtHandle_t handle = [] {
        hipblasLtHandle_t h = nullptr;
        auto status         = hipblasLtCreate(&h);
        if(status != HIPBLAS_STATUS_SUCCESS)
            h = nullptr;
        return h;
    }();
    return handle;
}

// Evaluate a dot instruction on the GPU using hipBLASLt directly.
// Falls back to returning an empty argument on failure.
static argument eval_dot_on_gpu(instruction_ref ins)
{
    // Get cached handle — if creation failed earlier, fall back immediately
    hipblasLtHandle_t handle = get_hipblaslt_handle();
    if(handle == nullptr)
        return {};

    // Evaluate inputs on CPU (lightweight ops like reshapes/contiguous on literals)
    std::vector<argument> input_args;
    for(const auto& input : ins->inputs())
    {
        auto arg = input->eval();
        if(arg.empty())
            return {};
        input_args.push_back(as_packed(arg));
    }
    if(input_args.size() < 2)
        return {};

    const auto& a_arg       = input_args[0];
    const auto& b_arg       = input_args[1];
    const auto& a_shape     = a_arg.get_shape();
    const auto& b_shape     = b_arg.get_shape();
    const auto output_shape = ins->get_shape();

    auto n_dim = output_shape.lens().size();
    if(n_dim < 2)
        return {};

    auto dim_0 = n_dim - 2;
    auto dim_1 = n_dim - 1;

    auto m = output_shape.lens()[dim_0];
    auto n = output_shape.lens()[dim_1];
    auto k = a_shape.lens()[dim_1];

    // Leading dimensions for packed row-major matrices
    auto lda = a_shape.strides()[dim_0]; // K for packed A[M,K]
    auto ldb = b_shape.strides()[dim_0]; // N for packed B[K,N]
    auto ldc = n;                        // N for packed C[M,N]

    // Batch dimensions
    std::size_t num_batches = 1;
    for(std::size_t i = 0; i < n_dim - 2; i++)
        num_batches *= output_shape.lens()[i];

    auto a_batch_stride =
        (n_dim > 2) ? static_cast<int64_t>(a_shape.strides()[n_dim - 3]) : int64_t{0};
    auto b_batch_stride =
        (n_dim > 2) ? static_cast<int64_t>(b_shape.strides()[n_dim - 3]) : int64_t{0};
    auto c_batch_stride = static_cast<int64_t>(m * n);

    auto hip_type = get_hipblaslt_type(a_shape.type());
    auto out_hip_type = get_hipblaslt_type(output_shape.type());
    auto compute_type =
        (hip_type == HIP_R_8I or hip_type == HIP_R_8U) ? HIPBLAS_COMPUTE_32I : HIPBLAS_COMPUTE_32F;
    auto scale_type =
        (compute_type == HIPBLAS_COMPUTE_32I) ? HIP_R_32I : HIP_R_32F;

    // Allocate GPU memory
    void* d_a = nullptr;
    void* d_b = nullptr;
    void* d_c = nullptr;

    if(hipMalloc(&d_a, a_shape.bytes()) != hipSuccess)
        return {};
    if(hipMalloc(&d_b, b_shape.bytes()) != hipSuccess)
    {
        (void)hipFree(d_a);
        return {};
    }
    auto out_packed_shape = output_shape.with_lens(output_shape.lens());
    if(hipMalloc(&d_c, out_packed_shape.bytes()) != hipSuccess)
    {
        (void)hipFree(d_a);
        (void)hipFree(d_b);
        return {};
    }

    // Copy input data to GPU
    (void)hipMemcpy(d_a, a_arg.data(), a_shape.bytes(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_b, b_arg.data(), b_shape.bytes(), hipMemcpyHostToDevice);

    // hipBLASLt uses column-major. For row-major C = A * B we compute
    // C^T = B^T * A^T, so the hipBLASLt "A" operand is our B and vice versa.
    hipblasLtMatrixLayout_t mat_a_layout = nullptr; // layout for our A (hipBLASLt "B")
    hipblasLtMatrixLayout_t mat_b_layout = nullptr; // layout for our B (hipBLASLt "A")
    hipblasLtMatrixLayout_t mat_c_layout = nullptr;

    // A^T in column-major: K x M, ld=lda
    hipblasLtMatrixLayoutCreate(&mat_a_layout, hip_type, k, m, lda);
    // B^T in column-major: N x K, ld=ldb
    hipblasLtMatrixLayoutCreate(&mat_b_layout, hip_type, n, k, ldb);
    // C^T in column-major: N x M, ld=ldc
    hipblasLtMatrixLayoutCreate(&mat_c_layout, out_hip_type, n, m, ldc);

    if(num_batches > 1)
    {
        int32_t batch_count = static_cast<int32_t>(num_batches);
        hipblasLtMatrixLayoutSetAttribute(
            mat_a_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(
            mat_b_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));
        hipblasLtMatrixLayoutSetAttribute(
            mat_c_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count));

        hipblasLtMatrixLayoutSetAttribute(mat_a_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &a_batch_stride,
                                          sizeof(a_batch_stride));
        hipblasLtMatrixLayoutSetAttribute(mat_b_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &b_batch_stride,
                                          sizeof(b_batch_stride));
        hipblasLtMatrixLayoutSetAttribute(mat_c_layout,
                                          HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                          &c_batch_stride,
                                          sizeof(c_batch_stride));
    }

    // Create matmul descriptor
    hipblasLtMatmulDesc_t matmul_desc = nullptr;
    hipblasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type);

    hipblasOperation_t op_n = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    hipblasLtMatmulDescSetAttribute(
        matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Get heuristic algorithm
    hipblasLtMatmulPreference_t preference = nullptr;
    hipblasLtMatmulPreferenceCreate(&preference);

    constexpr std::size_t max_workspace_bytes = 2 * 128 * 1024 * 1024;
    uint64_t max_ws                           = max_workspace_bytes;
    hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));

    hipblasLtMatmulHeuristicResult_t heuristic_result{};
    int returned_algo_count = 0;
    // hipBLASLt "A" = our B, "B" = our A (column-major transposition trick)
    auto heur_status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                       matmul_desc,
                                                       mat_b_layout,
                                                       mat_a_layout,
                                                       mat_c_layout,
                                                       mat_c_layout,
                                                       preference,
                                                       1,
                                                       &heuristic_result,
                                                       &returned_algo_count);

    auto cleanup = [&] {
        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatmulDescDestroy(matmul_desc);
        hipblasLtMatrixLayoutDestroy(mat_a_layout);
        hipblasLtMatrixLayoutDestroy(mat_b_layout);
        hipblasLtMatrixLayoutDestroy(mat_c_layout);
        (void)hipFree(d_a);
        (void)hipFree(d_b);
        (void)hipFree(d_c);
    };

    if(heur_status != HIPBLAS_STATUS_SUCCESS or returned_algo_count == 0)
    {
        cleanup();
        return {};
    }

    // Allocate workspace on GPU if needed
    void* d_workspace                  = nullptr;
    std::size_t actual_workspace_bytes = heuristic_result.workspaceSize;
    if(actual_workspace_bytes > 0)
    {
        if(hipMalloc(&d_workspace, actual_workspace_bytes) != hipSuccess)
        {
            cleanup();
            return {};
        }
    }

    float alpha_f   = 1.0f;
    float beta_f    = 0.0f;
    int32_t alpha_i = 1;
    int32_t beta_i  = 0;
    const void* alpha_ptr = (compute_type == HIPBLAS_COMPUTE_32I)
                                ? static_cast<const void*>(&alpha_i)
                                : static_cast<const void*>(&alpha_f);
    const void* beta_ptr = (compute_type == HIPBLAS_COMPUTE_32I)
                               ? static_cast<const void*>(&beta_i)
                               : static_cast<const void*>(&beta_f);

    // Run GEMM: hipBLASLt computes D = alpha * A_h * B_h + beta * C_h
    // where A_h = our B data, B_h = our A data (column-major trick)
    auto gemm_status = hipblasLtMatmul(handle,
                                       matmul_desc,
                                       alpha_ptr,
                                       d_b,          // hipBLASLt "A" = our B
                                       mat_b_layout, // layout for our B
                                       d_a,          // hipBLASLt "B" = our A
                                       mat_a_layout, // layout for our A
                                       beta_ptr,
                                       d_c,
                                       mat_c_layout,
                                       d_c,
                                       mat_c_layout,
                                       &heuristic_result.algo,
                                       d_workspace,
                                       actual_workspace_bytes,
                                       nullptr); // default stream

    (void)hipDeviceSynchronize();

    argument result;
    if(gemm_status == HIPBLAS_STATUS_SUCCESS)
    {
        // Copy result back to host via a literal so the data is properly owned
        std::vector<char> output_data(out_packed_shape.bytes());
        (void)hipMemcpy(output_data.data(), d_c, out_packed_shape.bytes(), hipMemcpyDeviceToHost);
        result = literal{out_packed_shape, output_data.data()}.get_argument();
    }

    if(d_workspace != nullptr)
        (void)hipFree(d_workspace);
    cleanup();
    return result;
}

#endif // MIGRAPHX_USE_HIPBLASLT

#if MIGRAPHX_USE_MIOPEN

static miopenDataType_t get_miopen_type(shape::type_t type)
{
    switch(type)
    {
    case shape::float_type: return miopenFloat;
    case shape::half_type: return miopenHalf;
    case shape::bf16_type: return miopenBFloat16;
    case shape::int32_type: return miopenInt32;
    case shape::int8_type: return miopenInt8;
    case shape::double_type:
    case shape::uint8_type:
    case shape::uint16_type:
    case shape::int16_type:
    case shape::int64_type:
    case shape::uint64_type:
    case shape::uint32_type:
    case shape::fp8e4m3fnuz_type:
    case shape::fp8e5m2fnuz_type:
    case shape::fp8e4m3fn_type:
    case shape::fp8e5m2_type:
    case shape::bool_type:
    case shape::tuple_type:
    case shape::fp4x2_type: break;
    }
    return miopenFloat;
}

// Thread-safe cached MIOpen handle. Created once on first use.
static miopenHandle_t get_miopen_handle()
{
    static miopenHandle_t handle = [] {
        miopenHandle_t h = nullptr;
        auto status      = miopenCreate(&h);
        if(status != miopenStatusSuccess)
            h = nullptr;
        return h;
    }();
    return handle;
}

// Evaluate a convolution instruction on the GPU using MIOpen immediate mode.
// Falls back to returning an empty argument on failure.
static argument eval_conv_on_gpu(instruction_ref ins)
{
    miopenHandle_t handle = get_miopen_handle();
    if(handle == nullptr)
        return {};

    // Get the convolution op parameters
    auto conv_op = any_cast<op::convolution>(ins->get_operator());

    // Evaluate inputs on CPU
    std::vector<argument> input_args;
    for(const auto& input : ins->inputs())
    {
        auto arg = input->eval();
        if(arg.empty())
            return {};
        input_args.push_back(as_packed(arg));
    }
    if(input_args.size() < 2)
        return {};

    const auto& x_arg   = input_args[0]; // input tensor
    const auto& w_arg   = input_args[1]; // weight tensor
    auto x_shape        = x_arg.get_shape();
    auto w_shape        = w_arg.get_shape();
    auto output_shape   = ins->get_shape();

    // MIOpen only supports float, half, bf16, int32, int8
    auto miopen_dtype = get_miopen_type(x_shape.type());
    if(x_shape.type() != shape::float_type and x_shape.type() != shape::half_type and
       x_shape.type() != shape::bf16_type and x_shape.type() != shape::int8_type and
       x_shape.type() != shape::int32_type)
        return {};

    // MIOpen requires at least 4D tensors (NCHW). Handle 1D conv (3D) by inserting dim.
    bool reshaped_1d = false;
    if(x_shape.ndim() == 3)
    {
        reshaped_1d   = true;
        auto x_lens   = x_shape.lens();
        auto w_lens   = w_shape.lens();
        auto out_lens = output_shape.lens();
        x_lens.insert(x_lens.begin() + 2, 1);
        w_lens.insert(w_lens.begin() + 2, 1);
        out_lens.insert(out_lens.begin() + 2, 1);
        x_shape      = shape{x_shape.type(), x_lens};
        w_shape      = shape{w_shape.type(), w_lens};
        output_shape = shape{output_shape.type(), out_lens};
    }

    // Create tensor descriptors
    auto make_tensor_desc = [&](const shape& s) -> miopenTensorDescriptor_t {
        miopenTensorDescriptor_t desc = nullptr;
        if(miopenCreateTensorDescriptor(&desc) != miopenStatusSuccess)
            return nullptr;
        auto ns = s.normalize_standard();
        std::vector<int> lens(ns.lens().begin(), ns.lens().end());
        std::vector<int> strides(ns.strides().begin(), ns.strides().end());
        if(miopenSetTensorDescriptor(
               desc, miopen_dtype, ns.lens().size(), lens.data(), strides.data()) !=
           miopenStatusSuccess)
        {
            miopenDestroyTensorDescriptor(desc);
            return nullptr;
        }
        return desc;
    };

    miopenTensorDescriptor_t x_desc = make_tensor_desc(x_shape);
    if(x_desc == nullptr)
        return {};
    miopenTensorDescriptor_t w_desc = make_tensor_desc(w_shape);
    if(w_desc == nullptr)
    {
        miopenDestroyTensorDescriptor(x_desc);
        return {};
    }
    auto out_packed_shape = output_shape.with_lens(output_shape.lens());
    miopenTensorDescriptor_t y_desc = make_tensor_desc(out_packed_shape);
    if(y_desc == nullptr)
    {
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        return {};
    }

    // Create convolution descriptor
    miopenConvolutionDescriptor_t conv_desc = nullptr;
    if(miopenCreateConvolutionDescriptor(&conv_desc) != miopenStatusSuccess)
    {
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }

    miopenConvolutionMode_t c_mode = miopenConvolution;
    if(conv_op.group > 1)
        c_mode = miopenGroupConv;

    int kdims = conv_op.kdims();
    // Ensure at least 2D for MIOpen
    if(reshaped_1d)
        kdims = 2;
    std::vector<int> pad_vec(std::max(2, kdims), 0);
    std::vector<int> stride_vec(std::max(2, kdims), 1);
    std::vector<int> dilation_vec(std::max(2, kdims), 1);

    std::copy_backward(
        conv_op.padding.begin(), conv_op.padding.begin() + conv_op.kdims(), pad_vec.end());
    std::copy_backward(conv_op.stride.begin(), conv_op.stride.end(), stride_vec.end());
    std::copy_backward(conv_op.dilation.begin(), conv_op.dilation.end(), dilation_vec.end());

    if(miopenInitConvolutionNdDescriptor(conv_desc,
                                         pad_vec.size(),
                                         pad_vec.data(),
                                         stride_vec.data(),
                                         dilation_vec.data(),
                                         c_mode) != miopenStatusSuccess)
    {
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }
    if(conv_op.group > 1)
        miopenSetConvolutionGroupCount(conv_desc, conv_op.group);

    // Use immediate mode: get heuristic solution (no benchmarking)
    size_t solution_count = 0;
    if(miopenConvolutionForwardGetSolutionCount(
           handle, w_desc, x_desc, conv_desc, y_desc, &solution_count) != miopenStatusSuccess or
       solution_count == 0)
    {
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }

    // Get the first (best heuristic) solution
    miopenConvSolution_t solution{};
    size_t returned_count = 0;
    if(miopenConvolutionForwardGetSolution(
           handle, w_desc, x_desc, conv_desc, y_desc, 1, &returned_count, &solution) !=
           miopenStatusSuccess or
       returned_count == 0)
    {
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }

    auto solution_id     = solution.solution_id;
    auto workspace_bytes = solution.workspace_size;

    // Allocate GPU memory
    void* d_x = nullptr;
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_workspace = nullptr;

    if(hipMalloc(&d_x, x_shape.bytes()) != hipSuccess)
    {
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }
    if(hipMalloc(&d_w, w_shape.bytes()) != hipSuccess)
    {
        (void)hipFree(d_x);
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }
    if(hipMalloc(&d_y, out_packed_shape.bytes()) != hipSuccess)
    {
        (void)hipFree(d_x);
        (void)hipFree(d_w);
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        return {};
    }
    if(workspace_bytes > 0)
    {
        if(hipMalloc(&d_workspace, workspace_bytes) != hipSuccess)
        {
            (void)hipFree(d_x);
            (void)hipFree(d_w);
            (void)hipFree(d_y);
            miopenDestroyConvolutionDescriptor(conv_desc);
            miopenDestroyTensorDescriptor(x_desc);
            miopenDestroyTensorDescriptor(w_desc);
            miopenDestroyTensorDescriptor(y_desc);
            return {};
        }
    }

    // Copy input data to GPU
    (void)hipMemcpy(d_x, x_arg.data(), x_arg.get_shape().bytes(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_w, w_arg.data(), w_arg.get_shape().bytes(), hipMemcpyHostToDevice);

    // Run convolution using immediate mode
    auto conv_status = miopenConvolutionForwardImmediate(handle,
                                                        w_desc,
                                                        d_w,
                                                        x_desc,
                                                        d_x,
                                                        conv_desc,
                                                        y_desc,
                                                        d_y,
                                                        d_workspace,
                                                        workspace_bytes,
                                                        solution_id);

    (void)hipDeviceSynchronize();

    argument result;
    if(conv_status == miopenStatusSuccess)
    {
        // Copy result back to host
        std::vector<char> output_data(out_packed_shape.bytes());
        (void)hipMemcpy(output_data.data(), d_y, out_packed_shape.bytes(), hipMemcpyDeviceToHost);
        // If we reshaped 1D → 4D, reshape back to 3D
        auto final_shape = reshaped_1d ? ins->get_shape().with_lens(ins->get_shape().lens())
                                       : out_packed_shape;
        result = literal{final_shape, output_data.data()}.get_argument();
    }

    // Cleanup
    if(d_workspace != nullptr)
        (void)hipFree(d_workspace);
    (void)hipFree(d_x);
    (void)hipFree(d_w);
    (void)hipFree(d_y);
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroyTensorDescriptor(x_desc);
    miopenDestroyTensorDescriptor(w_desc);
    miopenDestroyTensorDescriptor(y_desc);

    return result;
}

#endif // MIGRAPHX_USE_MIOPEN

void propagate_constant::apply(module& m) const
{
#if MIGRAPHX_USE_HIPBLASLT || MIGRAPHX_USE_MIOPEN
    // Pre-phase: find ALL constant convolution/dot instructions (including intermediate
    // ones buried inside transitive eval chains) and evaluate them on GPU before the
    // main constant-folding loop. This prevents expensive CPU eval() of convolutions
    // that are transitive dependencies of boundary constant instructions (e.g. an add
    // whose input is a constant convolution).
    // We iterate in program order (= topological order) and replace immediately so that
    // later ops in the chain see literals instead of expensive ops.
    if(enabled(MIGRAPHX_GPU_PROPAGATE{}))
    {
        std::vector<instruction_ref> gpu_const_ops;
        for(auto ins : iterator_for(m))
        {
            if(not is_const_ins(ins, skip_ops))
                continue;
            bool is_gpu_op = false;
#if MIGRAPHX_USE_HIPBLASLT
            if(ins->name() == "dot")
                is_gpu_op = true;
#endif
#if MIGRAPHX_USE_MIOPEN
            if(ins->name() == "convolution")
                is_gpu_op = true;
#endif
            if(is_gpu_op)
                gpu_const_ops.push_back(ins);
        }

        for(auto ins : gpu_const_ops)
        {
            argument gpu_result;
#if MIGRAPHX_USE_HIPBLASLT
            if(ins->name() == "dot")
            {
                const auto& out_s = ins->get_shape();
                const auto& a_s   = ins->inputs()[0]->get_shape();
                const auto& b_s   = ins->inputs()[1]->get_shape();
                std::cout << "[GPU propagate] pre-eval dot: A=" << a_s << " x B=" << b_s
                          << " -> C=" << out_s << " (" << out_s.bytes() / 1024 << " KB)"
                          << std::endl;
                gpu_result = eval_dot_on_gpu(ins);
            }
#endif
#if MIGRAPHX_USE_MIOPEN
            if(ins->name() == "convolution")
            {
                const auto& out_s = ins->get_shape();
                const auto& x_s   = ins->inputs()[0]->get_shape();
                const auto& w_s   = ins->inputs()[1]->get_shape();
                std::cout << "[GPU propagate] pre-eval conv: X=" << x_s << " W=" << w_s
                          << " -> Y=" << out_s << " (" << out_s.bytes() / 1024 << " KB)"
                          << std::endl;
                gpu_result = eval_conv_on_gpu(ins);
            }
#endif
            if(not gpu_result.empty())
            {
                std::cout << "[GPU propagate] pre-eval: replaced with literal" << std::endl;
                auto l = m.add_literal(gpu_result.get_shape(), gpu_result.data());
                m.replace_instruction(ins, l);
            }
            else
            {
                std::cout << "[GPU propagate] pre-eval: FAILED, will use CPU" << std::endl;
            }
        }
    }
#endif

    std::unordered_set<instruction_ref> const_instrs;
    auto last = std::prev(m.end());

    // Find instructions that can be evaluated to a literal
    for(auto i : iterator_for(m))
    {
        const bool is_const = is_const_ins(i, skip_ops);
        if(is_const and i != last)
            continue;

        if(i == last and is_const)
        {
            const_instrs.insert(i);
        }
        else
        {
            std::copy_if(i->inputs().begin(),
                         i->inputs().end(),
                         std::inserter(const_instrs, const_instrs.begin()),
                         [&](const instruction_ref ins) {
                             return is_const_ins(ins, skip_ops) and ins->name() != "@literal";
                         });
        }
    }

    // Compute literals in parallel
    std::vector<instruction_ref> const_instrs_vec{const_instrs.begin(), const_instrs.end()};
    std::vector<argument> literals(const_instrs_vec.size());
    std::size_t grainsize = 1;
#if !MIGRAPHX_HAS_EXECUTORS
    std::size_t n = std::max<std::size_t>(2048 / std::thread::hardware_concurrency(), 1);
    grainsize     = const_instrs_vec.size() / n;
#endif
    simple_par_for(const_instrs_vec.size(), grainsize, [&](const auto i) {
        if(enabled(MIGRAPHX_GPU_PROPAGATE{}))
        {
#if MIGRAPHX_USE_HIPBLASLT
            if(const_instrs_vec[i]->name() == "dot")
            {
                const auto& out_s = const_instrs_vec[i]->get_shape();
                const auto& a_s   = const_instrs_vec[i]->inputs()[0]->get_shape();
                const auto& b_s   = const_instrs_vec[i]->inputs()[1]->get_shape();
                std::cout << "[GPU propagate] dot: A=" << a_s << " x B=" << b_s
                          << " -> C=" << out_s
                          << " (" << out_s.bytes() / 1024 << " KB output)" << std::endl;
                auto gpu_result = eval_dot_on_gpu(const_instrs_vec[i]);
                if(not gpu_result.empty())
                {
                    std::cout << "[GPU propagate] dot: SUCCESS on GPU" << std::endl;
                    literals[i] = gpu_result;
                    return;
                }
                std::cout << "[GPU propagate] dot: FALLBACK to CPU" << std::endl;
            }
#endif
#if MIGRAPHX_USE_MIOPEN
            if(const_instrs_vec[i]->name() == "convolution")
            {
                const auto& out_s = const_instrs_vec[i]->get_shape();
                const auto& x_s   = const_instrs_vec[i]->inputs()[0]->get_shape();
                const auto& w_s   = const_instrs_vec[i]->inputs()[1]->get_shape();
                std::cout << "[GPU propagate] conv: X=" << x_s << " W=" << w_s
                          << " -> Y=" << out_s
                          << " (" << out_s.bytes() / 1024 << " KB output)" << std::endl;
                auto gpu_result = eval_conv_on_gpu(const_instrs_vec[i]);
                if(not gpu_result.empty())
                {
                    std::cout << "[GPU propagate] conv: SUCCESS on GPU" << std::endl;
                    literals[i] = gpu_result;
                    return;
                }
                std::cout << "[GPU propagate] conv: FALLBACK to CPU" << std::endl;
            }
#endif
        }
        literals[i] = as_packed(const_instrs_vec[i]->eval());
    });

    // Replace instructions in m
    for(size_t i = 0; i < const_instrs_vec.size(); i++)
    {
        if(not literals[i].empty())
        {
            if(enabled(MIGRAPHX_TRACE_PROPAGATE_CONSTANT{}))
            {
                std::cout << "Constant replace: " << std::endl;
                std::vector<instruction_ref> inss;
                fix([&](auto self, auto ins) {
                    if(contains(inss, ins))
                        return;
                    for(auto input : ins->inputs())
                        self(input);
                    inss.push_back(ins);
                })(const_instrs_vec[i]);
                m.debug_print(inss);
            }
            assert(literals[i].get_shape().lens() == const_instrs_vec[i]->get_shape().lens());
            assert(literals[i].get_shape().bytes() <= const_instrs_vec[i]->get_shape().bytes());
            auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
            m.replace_instruction(const_instrs_vec[i], l);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
