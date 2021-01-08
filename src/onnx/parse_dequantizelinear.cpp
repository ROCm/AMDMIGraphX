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

    int tune_axis(const int n_dim, const int axis) const
    {
        if(axis >= n_dim || axis < 0)
        {
            MIGRAPHX_THROW("DEQUANTIZELINEAR: axis is out of range.");
        }
        return (axis < 0) ? axis + n_dim : axis;
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto input_lens = args[0]->get_shape().lens();
        int n_dim = static_cast<int>(input_lens.size());

        auto sub_zero_point = args[0];

        if(args.size() == 3)
        {
            auto zero_point = args[2];
            if(not (zero_point->get_shape().elements() == 1))
            {
                axis = tune_axis(n_dim, axis);
                zero_point = info.add_instruction(
                    make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), zero_point);
            }
                
            auto zero_point_int8 = info.add_instruction(make_op("convert", {{"target_type", shape::int8_type}}), zero_point);
            auto sub_zero_point_int8 = info.add_instruction(make_op("convert", {{"target_type", shape::int8_type}}), sub_zero_point);
            sub_zero_point = info.add_broadcastable_binary_op("sub", sub_zero_point_int8, zero_point_int8);
        }

        auto dequant_input = info.add_instruction(
            make_op("convert", {{"target_type", shape::float_type}}), sub_zero_point);

        auto scale = args[1];
        if(not (scale->get_shape().elements() == 1))
        {
            axis = tune_axis(n_dim, axis);
            scale = info.add_instruction(
                make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), scale);
        }   
        return info.add_broadcastable_binary_op("mul", dequant_input, scale);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
