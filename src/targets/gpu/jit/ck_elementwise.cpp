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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const ck_elementwise_kernel = R"__migraphx__(
#include <migraphx/kernels/ck_elementwise.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace migraphx {

extern "C" {

__global__ void ck_elementwise_kernel(void* a_p, void* b_p, void* c_p) 
{
    using F16 = ck::half_t;
    using F32 = float;

    using ABDataType             = F16;
    using CDataType              = F16;
    using EltwiseComputeDataType = F32;

    using Add = ck::tensor_operation::element_wise::Add;

    using DeviceElementwiseAddInstance =
        ck::tensor_operation::device::DeviceBinaryElementwise<ABDataType,
                                                            ABDataType,
                                                            CDataType,
                                                            EltwiseComputeDataType,
                                                            Add,
                                                            1,
                                                            8,
                                                            8,
                                                            8,
                                                            8>;
    ck::index_t M = 1024;
    std::array<const void*, 2> input = {a_p,
                                        b_p};
    std::array<void*, 1> output      = {c_p};

    std::vector<ck::index_t> a_strides = {1};
    std::vector<ck::index_t> b_strides = {1};
    std::vector<ck::index_t> c_strides = {1};

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(
        input, output, {M}, {{a_strides}, b_strides}, {c_strides}, Add{});
}

}

} // namespace migraphx

)__migraphx__";

struct ck_elementwise_compiler : compiler<ck_elementwise_compiler>
{
    std::vector<std::string> names() const { return {"ck_elementwise"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "ck_elementwise_kernel";
        options.virtual_inputs = inputs;

        return compile_hip_code_object(ck_elementwise_kernel, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return replace(compile_op(ctx, to_shapes(ins->inputs()), op.to_value()));
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
