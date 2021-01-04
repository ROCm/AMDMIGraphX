#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_quantizelinear : op_parser<parse_quantizelinear>
{
    std::vector<op_desc> operators() const { return {{"QuantizeLinear"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                                       const onnx_parser& /*parser*/,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        auto quant_type = shape::uint8_type;
        int nargs       = args.size();

        if(nargs == 3)
            quant_type = args[2]->get_shape().type();
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto input_lens = args[0]->get_shape().lens();

        auto scale = args[1];
        if(axis != 1)
            scale = info.add_instruction(
                make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), scale);
        auto mul = info.add_broadcastable_binary_op("mul", args[0], scale);
        auto quantized =
            info.add_instruction(make_op("convert", {{"target_type", quant_type}}), mul);
        if(nargs == 3)
        {
            auto zero_point = args[2];
            if(axis != 1)
                zero_point = info.add_instruction(
                    make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), zero_point);
            quantized = info.add_broadcastable_binary_op("add", quantized, zero_point);
        }

        return quantized;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
