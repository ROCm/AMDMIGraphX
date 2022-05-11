#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_attention : op_parser<parse_attention>
{
    std::vector<op_desc> operators() const { return {{"Attention"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        auto input      = args[0];
        auto weights    = args[1];
        auto bias       = args[2];
        auto mask_index = args[3];

        instruction_ref past;
        instruction_ref extra_add_qk;
        bool is_past         = false;
        bool is_extra_add_qk = false;
        if(args.size() > 4)
        {
            past    = args[4];
            is_past = true;
        }
        if(args.size() == 6)
        {
            is_extra_add_qk = true;
            extra_add_qk    = args[5];
        }

        // ORT default is 12
        std::size_t num_heads = 12;
        if(contains(info.attributes, "num_heads"))
            num_heads = info.attributes.at("num_heads").i();

        // input shape: (batch_size, sequence_length, input_hidden_size)
        auto input_lens        = input->get_shape().lens();
        auto batch_size        = input_lens.at(0);
        auto sequence_length   = input_lens.at(1);
        auto input_hidden_size = input_lens.at(2);

        // bias shape: (3 * hidden_size)
        auto bias_lens           = bias->get_shape().lens();
        auto hidden_size         = bias_lens.at(0) / 3;
        auto head_size           = hidden_size / num_heads;
        int past_sequence_length = 0;

        // GetPresent
        //    Input and output shapes:
        //      past        : (2, batch_size, num_heads, past_sequence_length, head_size)
        //      present     : (2, batch_size, num_heads, past_sequence_length + sequence_length,
        //      head_size)
        std::vector<std::size_t> present_lens{2, batch_size, num_heads, sequence_length, head_size};

        if(is_past)
        {
            auto past_lens       = past->get_shape().lens();
            past_sequence_length = past_lens.at(3);
            present_lens[3] += past_lens[3];
        }

        // Use GEMM for fully connection.
        auto m = batch_size * sequence_length;
        auto n = bias_lens.front();
        auto k = input_hidden_size;

        // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
        auto bias_type = bias->get_shape().type();
        std::vector<float> ones_vec(m, 1);
        std::vector<std::size_t> ones_lens{1, m};
        auto ones =
            info.add_literal(migraphx::literal{migraphx::shape{bias_type, ones_lens}, ones_vec});
        bias        = info.add_instruction(migraphx::make_op("reshape", {{"dims", {n, 1}}}), bias);
        auto gemm_1 = info.add_instruction(
            migraphx::make_op("dot"),
            bias,
            ones /* info.make_contiguous(mb_bias), info.make_contiguous(ones) */);
        gemm_1 =
            info.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), gemm_1);

        /// ORT: Gemm, note that ROCM assumes col-major, so result(N, M) = 1 * weights x input + 1 x
        /// B. Assume row-major => results(N, M) = 1 * input x weights + 1 x B ?
        auto input_sq = info.add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch_size * sequence_length, hidden_size}}}),
            input);
        auto gemm_2    = info.add_instruction(migraphx::make_op("dot"), input_sq, weights);
        auto add_gemms = info.add_instruction(migraphx::make_op("add"), gemm_1, gemm_2);

        // LaunchAttentionKernel:
        //   LaunchTransQkv
        // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
        add_gemms = info.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims", {batch_size, sequence_length, 3, num_heads, head_size}}}),
            add_gemms);
        std::vector<std::size_t> qkv_perm{2, 0, 3, 1, 4};
        auto transqkv = info.add_instruction(
            migraphx::make_op("transpose", {{"permutation", qkv_perm}}), add_gemms);

        // now scratch3 has Q, K, V: each has size BxNxSxH
        // => transqkv has shape 3xBxNxSxH
        auto batches        = batch_size * num_heads;
        auto size_per_batch = sequence_length * head_size;
        auto total_size     = batches * size_per_batch;

        auto q_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transqkv);
        auto k_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), transqkv);
        auto v_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transqkv);
        q_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), q_t);
        k_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), k_t);
        v_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), v_t);

        if(is_past)
        {
            k_t = info.add_instruction(migraphx::make_op("concat", {{"axis", 3}}), past, k_t);
            v_t = info.add_instruction(
                migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {3}}}), k_t);
        }

        // Raw attention mask could be 2D (BxS) or 3D (BxSxS*) or 4D(Bx1xMxM), where M is the max
        // sequence length.
        auto mask_index_lens        = mask_index->get_shape().lens();
        bool use_raw_attention_mask = mask_index_lens.size() >= 2;

        // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
        // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
        const float rsqrt_head_size   = 1.f / sqrt(static_cast<float>(head_size));
        const int all_sequence_length = past_sequence_length + sequence_length;
        const int temp_matrix_size    = sequence_length * all_sequence_length;

        // For raw attention mask, the scalar if 1/sqrt(H) is moved to softmax computation.
        const float alpha = use_raw_attention_mask ? 1.0 : rsqrt_head_size;

        // K{B,N,S,H} -> K'{B,N,H,S}
        k_t = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k_t);
        auto gemm3 = info.add_instruction(migraphx::make_op("dot"), q_t, k_t);
        if(is_extra_add_qk)
            gemm3 = info.add_instruction(make_op("add"), gemm3, extra_add_qk);
        auto alpha_lit = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm3->get_shape().lens()}}),
            info.add_literal(
                migraphx::literal{migraphx::shape{gemm3->get_shape().type()}, {alpha}}));
        gemm3 =
            info.add_instruction(migraphx::make_op("mul"), gemm3, info.make_contiguous(alpha_lit));

        // apply softmax and store result P to scratch2: BxNxSxS*
        // Inference mask is all 1s => masking can be skipped
        auto softmax = info.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), gemm3);

        // compute P*V (as V*P), and store in scratch3: BxNxSxH
        auto gemm4 = info.add_instruction(migraphx::make_op("dot"), softmax, v_t);

        // scratch3 is BxNxSxH, transpose to output BxSxNxH
        gemm4 = info.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm4);
        gemm4 = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, sequence_length, num_heads * head_size}}}),
            info.make_contiguous(gemm4));
        return gemm4;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
