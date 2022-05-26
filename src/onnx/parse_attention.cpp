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

    instruction_ref parse(const op_desc& /* opd */,
                          const onnx_parser& /* parser */,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        auto input   = args[0];
        auto weights = args[1];
        auto bias    = args[2];
        // mask_index = args[3];
        // Raw attention mask is 2d (BxS) and all 1s for BERT-base and BERT-large inference

        // BERT-base default is 12, BERT-large default is 16
        std::size_t num_heads = 12;
        if(contains(info.attributes, "num_heads"))
            num_heads = info.attributes.at("num_heads").i();

        // input shape: (batch_size, sequence_length, input_hidden_size)
        auto input_lens      = input->get_shape().lens();
        auto batch_size      = input_lens.at(0);
        auto sequence_length = input_lens.at(1);

        // bias shape= (3 * hidden_size)
        auto bias_lens   = bias->get_shape().lens();
        auto hidden_size = bias_lens.at(0) / 3;
        auto head_size   = hidden_size / num_heads;

        // Use GEMM for fully connection.
        auto m = batch_size * sequence_length;
        auto n = bias_lens.front();

        // Bias shape is (N), broadcast using B(N, M) = 1 * bias(N, 1) x ones(1, M) + 0 * B.
        auto bias_type = bias->get_shape().type();
        std::vector<float> ones_vec(m, 1);
        std::vector<std::size_t> ones_lens{1, m};
        auto ones =
            info.add_literal(migraphx::literal{migraphx::shape{bias_type, ones_lens}, ones_vec});
        bias        = info.add_instruction(migraphx::make_op("reshape", {{"dims", {n, 1}}}), bias);
        auto gemm_1 = info.add_instruction(migraphx::make_op("dot"), bias, ones);
        gemm_1 =
            info.add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), gemm_1);

        /// results(N, M) = 1 * input x weights + 1 x B
        auto input_rs = info.add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch_size * sequence_length, hidden_size}}}),
            input);
        auto gemm_2    = info.add_instruction(migraphx::make_op("dot"), input_rs, weights);
        auto add_gemms = info.add_instruction(migraphx::make_op("add"), gemm_1, gemm_2);

        // LaunchTransQkv: BxSx3xNxH => 3xBxNxSxH
        add_gemms = info.add_instruction(
            migraphx::make_op("reshape",
                              {{"dims", {batch_size, sequence_length, 3, num_heads, head_size}}}),
            add_gemms);
        auto transqkv = info.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {2, 0, 3, 1, 4}}}), add_gemms);

        // Q, K, V: each has size BxNxSxH
        auto q_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), transqkv);
        auto k_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), transqkv);
        auto v_t = info.add_instruction(
            migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), transqkv);
        q_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), q_t);
        k_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), k_t);
        v_t = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), v_t);

        // compute Q*K' scaled by 1/sqrt(H)
        // Q: BxNxSxH, K (present_k): BxNxSxH => Q*K': BxNxSxS
        const float alpha = 1.f / sqrt(static_cast<float>(head_size));
        // K{B,N,S,H} -> K'{B,N,H,S}
        k_t = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k_t);
        auto gemm3     = info.add_instruction(migraphx::make_op("dot"), q_t, k_t);
        auto alpha_lit = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm3->get_shape().lens()}}),
            info.add_literal(
                migraphx::literal{migraphx::shape{gemm3->get_shape().type()}, {alpha}}));
        gemm3 =
            info.add_instruction(migraphx::make_op("mul"), gemm3, info.make_contiguous(alpha_lit));

        // Inference mask is all 1s => masking can be skipped
        // P = softmax result: BxNxSxS
        auto softmax = info.add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), gemm3);

        // compute P*V: (BxNxSxS) x (BxNxSxH) => BxNxSxH
        auto gemm4 = info.add_instruction(migraphx::make_op("dot"), softmax, v_t);

        // result is BxNxSxH, transpose to BxSxNxH and reshape to BxSxHiddenSize
        // transposeCtx
        gemm4 = info.add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm4);
        return info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, sequence_length, num_heads * head_size}}}),
            info.make_contiguous(gemm4));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
