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
#include <fstream>
#include <migraphx/filesystem.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>

#include <migraphx/gpu/ck.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reduce_dims.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

struct RotaryParameters
{
    float scale;
    int batch_size;           // Batch size used by input
    int sequence_length;      // Sequence length used by input
    int hidden_size;          // Hidden size used by input
    int head_size;            // Head size
    int rotary_embedding_dim; // Rotary embedding dimension.
    int num_heads;            // num_heads = hidden_size / head_size
    int max_sequence_length;  // Sequence length used by cos/sin cache
    int head_stride;          // Head stride
    int seq_stride;           // Sequence stride
    int batch_stride;         // Batch stride
    int position_ids_format;  // Format of position ids - 0 is (1), 1 is (batch_size,
                              // sequence_length)
    bool transposed; // Whether the input tensor has been transposed into (batch, num_heads,
                     // seq_len, hidden)
    int seqlen_present_kv_cache;

    int do_rotary;
    int kv_num_heads;
    int local_window_size;
    int rotary_interleaved;

    std::string make_init_str()
    {
        std::string str =   std::to_string(scale) + ", " +
                            std::to_string(batch_size) + ", " +
                            std::to_string(sequence_length) + ", " +
                            std::to_string(hidden_size) + ", " +
                            std::to_string(head_size) + ", " +
                            std::to_string(rotary_embedding_dim) + ", " +
                            std::to_string(num_heads) + ", " +
                            std::to_string(max_sequence_length) + ", " +
                            std::to_string(head_stride) + ", " +
                            std::to_string(seq_stride) + ", " +
                            std::to_string(batch_stride) + ", " +
                            std::to_string(position_ids_format) + ", " +
                            std::to_string(transposed) + ", " +
                            std::to_string(seqlen_present_kv_cache) + ", " +
                            std::to_string(do_rotary) + ", " +
                            std::to_string(kv_num_heads) + ", " +
                            std::to_string(local_window_size) + ", " +
                            std::to_string(rotary_interleaved);
        return str;
    }
    
};


using namespace migraphx::gpu::gen; // NOLINT

// NOLINTNEXTLINE
static const char* const group_query_attention_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/group_query_attention.hpp>
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/ops.hpp>


namespace migraphx {



extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors(), rotate_last())(${args})([](auto... xs) {
        
        group_query_attention_matrix(xs..., make_rotary_params(${rotary_params}));
    });
}

}

} // namespace migraphx

)__migraphx__";

struct group_query_attention_compiler : compiler<group_query_attention_compiler>
{
    std::vector<std::string> names() const { return {"group_query_attention", "gpu::group_query_attention"}; }

    // ck::host::device_gemm_multiple_d::Problem create_problem(std::size_t m, std::size_t n, std::size_t k, bool trans_b, shape::type_t dtype) const
    // {
    //     std::string ck_passthrough = "ck_passthrough";
    //     return ck::host::device_gemm_multiple_d::Problem{m,
    //                                                      n,
    //                                                      k,
    //                                                      false,
    //                                                      trans_b,
    //                                                      false,
    //                                                      {},
    //                                                      get_type(dtype),
    //                                                      get_type(dtype),
    //                                                      get_type(dtype),
    //                                                      {},
    //                                                      ck_passthrough,
    //                                                      ck_passthrough,
    //                                                      ck_passthrough};
    // }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        assert(v.contains("num_heads"));
        auto num_heads = v.at("num_heads").to<int>();
        assert(v.contains("kv_num_heads"));
        auto kv_num_heads = v.at("kv_num_heads").to<int>();


        auto q_shape              = inputs[0];
        auto q_lens               = q_shape.lens();
        auto input_type = q_shape.type();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[2];
        std::size_t head_size     = q_lens[3];
        auto q_hidden_size = q_lens[1] * head_size;
        const bool packed_qkv = true;

        std::size_t rotary_dim = inputs[5].lens()[1] * 2;
        std::size_t present_kv_seqlen = 4096;

        auto seq_stride  = head_size;
        auto head_stride = sequence_length * seq_stride;
        auto batch_stride =
                        (packed_qkv ? (num_heads + 2 * kv_num_heads) : num_heads) * head_stride;
        auto position_ids_format = sequence_length == 1 ? 1 : 0;
        bool transposed          = true;

        RotaryParameters rotary_params;
        rotary_params.batch_size           = batch_size;
        rotary_params.sequence_length      = sequence_length;
        rotary_params.hidden_size          = q_hidden_size;
        rotary_params.head_size            = head_size;
        rotary_params.rotary_embedding_dim = rotary_dim;
        rotary_params.num_heads            = num_heads;
        rotary_params.max_sequence_length  = sequence_length; // unused
        rotary_params.seq_stride           = head_size;
        rotary_params.head_stride          = sequence_length * rotary_params.seq_stride;
        rotary_params.batch_stride         = batch_stride;
        rotary_params.position_ids_format = position_ids_format;
        rotary_params.transposed          = transposed;
        rotary_params.seqlen_present_kv_cache = present_kv_seqlen;
        assert(v.contains("do_rotary"));
        auto do_rotary = v.at("do_rotary").to<int>();
        assert(v.contains("local_window_size"));
        auto local_window_size = v.at("local_window_size").to<int>();
        assert(v.contains("rotary_interleaved"));
        auto rotary_interleaved = v.at("rotary_interleaved").to<int>();
        assert(v.contains("scale"));
        auto scale = v.at("scale").to<float>();
        rotary_params.do_rotary = do_rotary;
        rotary_params.kv_num_heads = kv_num_heads;
        rotary_params.local_window_size = local_window_size;
        rotary_params.rotary_interleaved = rotary_interleaved;
        rotary_params.scale = scale;



        auto rotary_params_str = rotary_params.make_init_str();

        auto virtual_inputs = inputs;
        hip_compile_options options;
        // for(auto i = 0; i < inputs.size(); ++i)
        // {
        //     std::cout << inputs[i] << std::endl;
        // }
        //options.additional_src_files = ck_headers();
        // auto grid_size = batch_size * blocks_per_batch;
        // options.set_launch_params(v, grid_size * block_size, block_size);
        options.set_launch_params(v, compute_global_for(ctx, inputs.at(0).elements() * 4096));
        int blocks_per_batch = 1; ////
        options.inputs         = virtual_inputs;
        options.output         = inputs.back();//output_shape;
        options.kernel_name    = v.get("kernel", "group_query_attention_kernel");
        // options.virtual_inputs = virtual_inputs;

        if(v.get("check", false) or enabled(MIGRAPHX_CK_DEBUG{}))
            options.emplace_param("-DMIGRAPHX_CK_CHECK=1");

        // std::cout << rotary_params_str << std::endl;
        auto src = interpolate_string(group_query_attention_kernel,
                                      {
                                    //    {"solution", template_str},
                                    //    {"include", include_header},
                                       {"params", enum_params(virtual_inputs.size(), "void * private_p")},
                                       {"args", enum_params(virtual_inputs.size(), "private_p")},
                                       {"blocks_per_batch", to_string(blocks_per_batch)},
                                       {"rotary_params", rotary_params_str},
                                       //{"preamble", v.get("preamble", std::string{})},
                                       {"kernel", options.kernel_name}});
        // std::cout << "compile hip..." << std::endl;
        return compile_hip_code_object(src, options);
    }

    // value create_settings(instruction_ref ins, const operation& op) const
    // {
    //     auto v      = op.to_value();
    //     v["kernel"] = "group_query_attention_kernel";
    //     return v;
    // }

    compiler_replace
    compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        // std::cout << "to shapes..." << std::endl;
        auto shapes = to_shapes(ins->inputs());
        // std::cout << "to value..." << std::endl;
        auto v = op.to_value();
        // std::cout << "compile_op..." << std::endl;
        // auto v      = create_settings(ins, op);
        // if(not solution.is_null())
        //     v["tuning_value"] = solution;
        return compile_op(ctx, shapes, v);
        // {compile_op(ctx, shapes, v),
        //         [=](module& m, instruction_ref ins2, const operation& code_object) {
        //             if(enabled(MIGRAPHX_LOG_CK_GEMM{}))
        //             {
        //                 std::vector<shape> gemm_shapes{
        //                     shapes[0], shapes[1], shapes.back().with_type(shapes[0].type())};
        //                 std::cout << "gpu::group_query_attention: " << to_json_string(to_value(gemm_shapes))
        //                           << std::endl;
        //             }
        //             m.replace_instruction(ins2, code_object, ins2->inputs());
        //         }};
    }

    // optional<tuning_config>
    // get_tuning_config(context& ctx, instruction_ref ins, const operation& op, bool exhaustive) const
    // {
    //     if(not exhaustive and not enabled(MIGRAPHX_TUNE_CK{}))
    //         return nullopt;
    //     tuning_config tc;
    //     auto shapes    = to_shapes(ins->inputs());
    //     auto problem   = create_problem(shapes, create_settings(ins, op));
    //     auto solutions = problem.GetSolutions(ctx.get_current_device().get_gfx_name());
    //     tc.solutions.resize(solutions.size());
    //     std::iota(tc.solutions.begin(), tc.solutions.end(), 0);
    //     std::vector<shape> gemm_shapes{shapes[0], shapes[1], shapes.back()};
    //     tc.problem = to_value(gemm_shapes);
    //     return tc;
    // }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
