#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_dequantizelinear : op_parser<parse_dequantizelinear>
{
    std::vector<op_desc> operators() const { return {{"DequantizeLinear"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto x       = args[0];
        auto x_scale = args[1];

        if(args.size() == 3)
        {
            auto x_zero_point = args[2];
            return info.add_instruction(
                make_op("dequantizelinear", {{"axis", axis}}), x, x_scale, x_zero_point);
        }

        auto x_zero_point = info.add_literal(0);
        return info.add_instruction(
            make_op("dequantizelinear", {{"axis", axis}}), x, x_scale, x_zero_point);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
