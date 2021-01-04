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

        auto input_lens = args[0]->get_shape().lens();

        auto sub_zero_point = args[0];

        if(args.size() == 3)
        {
            auto zero_point = args[2];
            if(axis != 1)
                zero_point = info.add_instruction(
                    make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), zero_point);
            sub_zero_point = info.add_broadcastable_binary_op("sub", sub_zero_point, zero_point);
        }

        auto dequant_input = info.add_instruction(
            make_op("convert", {{"target_type", shape::float_type}}), sub_zero_point);

        auto scale = args[1];
        if(axis != 1)
            scale = info.add_instruction(
                make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), scale);
        return info.add_broadcastable_binary_op("mul", dequant_input, scale);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
