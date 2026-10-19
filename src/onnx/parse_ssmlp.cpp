/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/matmulnbits.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

namespace {

// SkipSimplifiedLayerNormalization core: returns {normalized, input_skip_sum}.
std::pair<instruction_ref, instruction_ref>
skip_simplified_ln(onnx_parser::node_info& info,
                   instruction_ref x,
                   instruction_ref skip,
                   instruction_ref gamma,
                   float epsilon)
{
    auto x_dtype = x->get_shape().type();
    int64_t axis = static_cast<int64_t>(x->get_shape().ndim()) - 1;

    auto skip_sum = info.add_common_op("add", x, skip);
    auto float_x  = info.add_instruction(
        make_op("convert", {{"target_type", migraphx::shape::float_type}}), skip_sum);
    auto x_sq = info.add_common_op("mul", float_x, float_x);
    auto rms  = info.add_instruction(make_op("reduce_mean", {{"axes", {axis}}}), x_sq);
    rms       = info.add_instruction(make_op("convert", {{"target_type", x_dtype}}), rms);
    epsilon =
        (x_dtype == migraphx::shape::half_type and std::abs(epsilon) < 1e-7) ? 1e-7 : epsilon;
    auto eps  = info.add_literal(migraphx::literal{migraphx::shape{x_dtype}, {epsilon}});
    rms       = info.add_common_op("add", rms, eps);
    auto rrms = info.add_instruction(make_op("rsqrt"), rms);
    auto result = info.add_common_op("mul", skip_sum, rrms);
    result      = info.add_common_op("mul", result, gamma);
    return {result, skip_sum};
}

instruction_ref silu(onnx_parser::node_info& info, instruction_ref x)
{
    auto sig = info.add_instruction(make_op("sigmoid"), x);
    return info.add_common_op("mul", x, sig);
}

} // namespace

// com.ryzenai.SSMLP: SkipSimplifiedLN + SwiGLU MLP (3x MatMulNBits) + next SkipSimplifiedLN.
// Inputs:
//   0 residual, 1 attn_out, 2 post_attn_ln_gamma,
//   3-5 gate qweight/scales/qzeros, 6-8 up, 9-11 down,
//   12 next_input_ln_gamma, 13-15 packed (ignored)
// Outputs: next_ln_out, next_skip_sum
struct parse_ssmlp : op_parser<parse_ssmlp>
{
    std::vector<op_desc> operators() const { return {{"SSMLP"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       const std::vector<instruction_ref>& args) const
    {
        if(args.size() < 13)
            MIGRAPHX_THROW("SSMLP: expects at least 13 inputs, got " + std::to_string(args.size()));

        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();

        auto get_i = [&](const std::string& name) {
            if(not contains(info.attributes, name))
                MIGRAPHX_THROW("SSMLP: Attribute " + name + " required, but is missing");
            return parser.parse_value(info.attributes.at(name)).at<int>();
        };

        auto [mlp_in, residual] =
            skip_simplified_ln(info, args[1], args[0], args[2], epsilon);

        auto gate = matmulnbits::add_matmul_nbits(info,
                                                  mlp_in,
                                                  args[3],
                                                  args[4],
                                                  args[5],
                                                  std::nullopt,
                                                  get_i("gate_N"),
                                                  get_i("gate_K"),
                                                  get_i("gate_bits"),
                                                  get_i("gate_block_size"));
        auto up   = matmulnbits::add_matmul_nbits(info,
                                                mlp_in,
                                                args[6],
                                                args[7],
                                                args[8],
                                                std::nullopt,
                                                get_i("up_N"),
                                                get_i("up_K"),
                                                get_i("up_bits"),
                                                get_i("up_block_size"));
        auto mid  = info.add_common_op("mul", silu(info, gate), up);
        auto down = matmulnbits::add_matmul_nbits(info,
                                                  mid,
                                                  args[9],
                                                  args[10],
                                                  args[11],
                                                  std::nullopt,
                                                  get_i("down_N"),
                                                  get_i("down_K"),
                                                  get_i("down_bits"),
                                                  get_i("down_block_size"));

        auto [out, skip_sum] = skip_simplified_ln(info, down, residual, args[12], epsilon);
        return {out, skip_sum};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
