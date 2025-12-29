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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Kernel template for multi-source gather → slice → concat
// 
// For each output position (batch, seq, col):
//   1. Find which source owns this column
//   2. Get the gather row index for that source
//   3. Look up indices[source][row, seq] to get embedding row
//   4. Read embedding[source][emb_row, local_col]
//
// NOLINTNEXTLINE
static const char* const multi_gather_concat_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <args.hpp>

namespace migraphx {

// Source metadata - generated at compile time
constexpr index_int num_sources = ${num_sources};
__device__ constexpr index_int row_indices[${num_sources}] = {${row_indices_str}};
__device__ constexpr index_int col_offsets[${num_sources}] = {${col_offsets_str}};

// Find which source owns the given output column
__device__ inline index_int find_source(index_int col)
{
    // Iterate forwards, find the last source where col >= col_offsets[s]
    index_int result = 0;
    for(index_int s = 0; s < num_sources; s++)
    {
        if(col >= col_offsets[s])
            result = s;
    }
    return result;
}

extern "C" {

// Multi-gather-concat kernel
// Inputs: emb0, idx0, emb1, idx1, ..., output
MIGRAPHX_GLOBAL void multi_gather_concat_kernel(${param_list})
{
    make_tensors()(${arg_list})([](${tensor_params}, auto out) {
        auto ind = make_index();
        auto nelem = out.get_shape().elements();
        auto out_shape = out.get_shape();
        
        ind.global_stride(nelem, [&](auto i) {
            auto out_idx = out_shape.multi(i);
            
            // out_idx = [batch, seq, col]
            // batch is always 0 (after slice), seq comes from indices, col is the concat dim
            index_int seq = out_idx[1];
            index_int col = out_idx[2];
            
            // Find which source this column belongs to
            index_int src = find_source(col);
            index_int local_col = col - col_offsets[src];
            index_int gather_row = row_indices[src];
            
            // Get the embedding row from indices
            // Each indices tensor has shape [N, seq_len]
            ${source_switch}
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

struct multi_gather_concat_compiler : compiler<multi_gather_concat_compiler>
{
    std::vector<std::string> names() const { return {"multi_gather_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        options.inputs         = inputs;
        options.output         = out_s;
        options.virtual_inputs = inputs;
        options.kernel_name    = "multi_gather_concat_kernel";

        // Extract metadata from op value
        auto row_indices = v.at("row_indices").to_vector<int64_t>();
        auto col_offsets = v.at("col_offsets").to_vector<int64_t>();
        auto num_sources = v.at("num_sources").to<int64_t>();

        // Build array initializer strings
        auto to_array_str = [](const std::vector<int64_t>& vec) {
            std::string s;
            for(size_t i = 0; i < vec.size(); i++)
            {
                if(i > 0) s += ", ";
                s += std::to_string(vec[i]);
            }
            return s;
        };

        // Build parameter list: p_emb0, p_idx0, ... (outer void* params)
        // Lambda params use different names: e0, i0, ... to avoid shadowing
        std::string param_list;
        std::string arg_list;
        std::string tensor_params;
        for(int64_t s = 0; s < num_sources; s++)
        {
            if(s > 0)
            {
                param_list += ", ";
                arg_list += ", ";
                tensor_params += ", ";
            }
            // Outer function params use p_ prefix
            param_list += "void* p_emb" + std::to_string(s) + ", void* p_idx" + std::to_string(s);
            arg_list += "p_emb" + std::to_string(s) + ", p_idx" + std::to_string(s);
            // Lambda params use e/i prefix (no shadowing)
            tensor_params += "auto e" + std::to_string(s) + ", auto i" + std::to_string(s);
        }
        param_list += ", void* p_output";
        arg_list += ", p_output";

        // Build the source switch statement
        // This generates code to read from the correct embedding based on source index
        std::string source_switch;
        for(int64_t s = 0; s < num_sources; s++)
        {
            if(s == 0)
                source_switch += "if(src == 0) {\n";
            else
                source_switch += "} else if(src == " + std::to_string(s) + ") {\n";
            
            source_switch += "    auto idx_shape" + std::to_string(s) + " = i" + std::to_string(s) + ".get_shape();\n";
            source_switch += "    auto idx_pos" + std::to_string(s) + " = idx_shape" + std::to_string(s) + ".multi(0);\n";
            source_switch += "    idx_pos" + std::to_string(s) + "[0] = gather_row;\n";
            source_switch += "    idx_pos" + std::to_string(s) + "[1] = seq;\n";
            source_switch += "    auto emb_row = static_cast<int64_t>(i" + std::to_string(s) + "[idx_pos" + std::to_string(s) + "]);\n";
            source_switch += "    auto emb_dim0 = e" + std::to_string(s) + ".get_shape().lens[0];\n";
            source_switch += "    if(emb_row < 0) emb_row += static_cast<int64_t>(emb_dim0);\n";
            source_switch += "    if(emb_row < 0) emb_row = 0;\n";
            source_switch += "    if(emb_row >= static_cast<int64_t>(emb_dim0)) emb_row = static_cast<int64_t>(emb_dim0) - 1;\n";
            source_switch += "    auto emb_shape" + std::to_string(s) + " = e" + std::to_string(s) + ".get_shape();\n";
            source_switch += "    auto emb_pos" + std::to_string(s) + " = emb_shape" + std::to_string(s) + ".multi(0);\n";
            source_switch += "    emb_pos" + std::to_string(s) + "[0] = static_cast<index_int>(emb_row);\n";
            source_switch += "    emb_pos" + std::to_string(s) + "[1] = static_cast<index_int>(local_col);\n";
            source_switch += "    out[i] = e" + std::to_string(s) + "[emb_pos" + std::to_string(s) + "];\n";
        }
        source_switch += "}\n";

        auto src = interpolate_string(multi_gather_concat_kernel,
                                      {{"num_sources", std::to_string(num_sources)},
                                       {"row_indices_str", to_array_str(row_indices)},
                                       {"col_offsets_str", to_array_str(col_offsets)},
                                       {"param_list", param_list},
                                       {"arg_list", arg_list},
                                       {"tensor_params", tensor_params},
                                       {"source_switch", source_switch}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

