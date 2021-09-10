#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_thresholdedrelu : op_parser<parse_thresholdedrelu>
{
    std::vector<op_desc> operators() const { return {{"ThresholdedRelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        float alpha = 1.0;
        if(contains(info.attributes, "alpha"))
            alpha = parser.parse_value(info.attributes.at("alpha")).at<float>();

        auto x_shape = args[0]->get_shape();

        auto lit_zero = info.add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {0}});
        auto lit_alpha =
            info.add_literal(migraphx::literal{migraphx::shape{x_shape.type()}, {alpha}});
        auto mb_zero = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_shape.lens()}}), lit_zero);
        auto mb_alpha = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_shape.lens()}}), lit_alpha);
        auto condition = info.add_instruction(migraphx::make_op("greater"), args[0], mb_alpha);

        return info.add_instruction(migraphx::make_op("where"), condition, args[0], mb_zero);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
