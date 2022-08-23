#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_layernorm : op_parser<parse_layernorm>
{
    std::vector<op_desc> operators() const { return {{"LayerNormalization"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        // un-fuse layernorm op so migraphx can handle fusion instead

        auto x       = args.front();
        auto x_type  = x->get_shape().type();
        auto weights = args.at(1);
        auto bias    = args.at(2);

        float epsilon = 1e-12f;
        int64_t axis  = -1;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int64_t>();
        }

        auto epsilon_lit = info.add_literal(literal{shape{x_type, {1}}, {epsilon}});
        auto exponent    = info.add_literal(literal{shape{x_type, {1}}, {2.0}});
        auto dims        = x->get_shape().lens();

        auto mean = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {axis}}}), x);
        auto mean_mbcast =
            info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", dims}}), mean);
        auto sub             = info.add_instruction(migraphx::make_op("sub"), x, mean_mbcast);
        auto exponent_mbcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dims}}), exponent);
        auto pow = info.add_instruction(migraphx::make_op("pow"), sub, exponent_mbcast);
        auto var = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {axis}}}), pow);
        auto add_epsilon = info.add_broadcastable_binary_op("add", var, epsilon_lit);
        auto sqrt        = info.add_instruction(migraphx::make_op("sqrt"), add_epsilon);
        auto div         = info.add_broadcastable_binary_op("div", sub, sqrt);
        auto mul         = info.add_broadcastable_binary_op("mul", div, weights);

        return info.add_broadcastable_binary_op("add", mul, bias);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
