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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
// #include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <amdmlss/amdmlss_api.h>
#include <migraphx/gpu/code_object_op.hpp>

#include <cctype>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

void checkStatus(MLSSstatus status, int line)
{
    if (status != MLSS_SUCCESS)
    {
        MLSSstring err = mlssGetErrorString(status);

        printf("Failed at line: %d, :%s\n", line, err);
        free(err);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_STATUS(status) checkStatus((status), __LINE__)

struct mlss_compiler : compiler<mlss_compiler>
{
    std::vector<std::string> names() const { return {"mlss_mha"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {

        const auto& device = ctx.get_current_device();
        std::string target_arch = device.get_device_name();
        std::size_t num_cu = device.get_cu_count();

        auto query_dim  = inputs[0].ndim();
        auto query_lens = inputs[0].lens();

        auto head_size_v = query_lens[4];

        MLSScontext context = 0;

        for (char &ch : target_arch) {
            ch = std::toupper(ch);
        }
        target_arch = "MLSS_" + target_arch;

        MLSSstring asic = target_arch.data();
        MLSSstring opName = MLSS_MHA;

        


        MLSSuint32 batch_size = query_lens[0];
        MLSSuint32 head_num = 8;
        MLSSuint32 q_sequence_length = query_lens[1];
        MLSSuint32 kv_sequence_length = query_lens[1];;
        MLSSuint32 head_dim = query_lens[4];
        MLSSuint32 packing = 0;
        float scale = 1 / std::sqrt(head_dim);
        MLSSenum data_type = MLSS_FLOAT16;
        MLSSuint32 kvDim = 0;

        CHECK_STATUS(mlssCreateContext(&context, asic, opName));

        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_BATCH, &batch_size));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_QSEQ, &q_sequence_length));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_KVSEQ, &kv_sequence_length));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_KDIM, &kvDim));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_VDIM, &kvDim));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_SIZEHEADS, &head_dim));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_PACKING, &packing));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_HEADCOUNT, &head_num));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_SCALE, &scale));
        CHECK_STATUS(mlssSetParameterByEnum(&context, opName, MLSS_ATTR_MHA_DATATYPE, &data_type));
        
        MLSSstatus* pStatuses = NULL;
        MLSSsize nStatuses = 0;

        if (mlssGetCaps(context, &pStatuses, &nStatuses) != MLSS_SUCCESS)
        {
            std::cout << "Failed to get caps\n" << std::endl;
        }
        else
        {
            std::cout << "Got caps\n" << std::endl;
        }
        
        MLSSbinary* binaries = NULL;
        MLSSsize n = 0;
        CHECK_STATUS(mlssGetBinaries(context, &binaries, &n));
        CHECK_STATUS(mlssPrintBinaries(binaries, n));

        const auto& binary = binaries[0];

        std::string kernel_name = (binary.m_pKernelName ? binary.m_pKernelName : "N/A");
        size_t bin_size = binary.m_binarySize;

        value::binary value_binary((char*)binary.m_binaries, bin_size);

        auto nelements  = inputs.back().elements();
        auto block_size = compute_block_size(ctx, nelements, 256);
        hip_compile_options options;
        options.set_launch_params(
            v, compute_global_for(ctx, nelements * block_size, 256), block_size);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = kernel_name;

        return code_object_op{value_binary,
                          kernel_name,
                          options.global,
                          options.local,
                          options.inputs,
                          options.output,
                          options.output_arg};
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
