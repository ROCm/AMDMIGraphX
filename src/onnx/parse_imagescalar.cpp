#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_imagescalar : op_parser<parse_imagescalar>
{
    std::vector<op_desc> operators() const { return {{"ImageScaler"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        float scale = 1.0;
        std::vector<float> bias{};
        if(contains(info.attributes, "scale"))
        {
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }

        if(contains(info.attributes, "bias"))
        {
            auto&& bias_floats = info.attributes["bias"].floats();
            bias               = std::vector<float>(bias_floats.begin(), bias_floats.end());
        }
        auto input_shape       = args.front()->get_shape();
        auto const& input_lens = input_shape.lens();
        auto input_type        = input_shape.type();

        auto scale_val = info.add_literal(literal{shape{input_type}, {scale}});
        auto bias_vals = info.add_literal(literal{shape{input_type, {bias.size()}}, bias});

        auto scale_tensor = info.add_instruction(
            migraphx::make_op("scalar", {{"scalar_bcst_dims", input_lens}}), scale_val);
        auto img_scaled =
            info.add_instruction(migraphx::make_op("mul"), args.front(), scale_tensor);
        auto bias_bcast = info.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", input_lens}}), bias_vals);
        return info.add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
