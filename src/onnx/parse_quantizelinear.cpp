#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_quantizelinear : op_parser<parse_quantizelinear>
{
    std::vector<op_desc> operators() const { return {{"QuantizeLinear"}}; }

    // y = saturate(round(x / y_scale) + zero_point)
    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto quant_type = shape::uint8_type;
        int nargs       = args.size();

        int max_quant = 255;
        int min_quant = 0;

        if(nargs == 3)
            quant_type = args[2]->get_shape().type();

        if(quant_type == shape::int8_type)
        {
            max_quant = 127;
            min_quant = -128;
        }

        auto max_arg = info.add_literal(max_quant);
        auto min_arg = info.add_literal(min_quant);

        int axis = 1;

        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto input_lens = args[0]->get_shape().lens();
        int n_dim       = static_cast<int>(input_lens.size());

        auto scale = args[1];
        if(not(scale->get_shape().elements() == 1))
        {
            axis  = tune_axis(n_dim, axis, opd.op_name);
            scale = info.add_instruction(
                make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), scale);
        }

        auto div            = info.add_broadcastable_binary_op("div", args[0], scale);
        auto div_round      = info.add_instruction(make_op("round"), div);
        auto add_zero_point = div_round;

        if(nargs == 3)
        {
            auto zero_point = args[2];
            if(not(zero_point->get_shape().elements() == 1))
            {
                axis       = tune_axis(n_dim, axis, opd.op_name);
                zero_point = info.add_instruction(
                    make_op("broadcast", {{"axis", axis}, {"dims", input_lens}}), zero_point);
            }

            zero_point = info.add_instruction(
                make_op("convert", {{"target_type", shape::int32_type}}), zero_point);
            add_zero_point = info.add_instruction(
                make_op("convert", {{"target_type", shape::int32_type}}), add_zero_point);
            add_zero_point = info.add_broadcastable_binary_op("add", add_zero_point, zero_point);
        }

        auto s           = add_zero_point->get_shape();
        const auto& lens = s.lens();
        std::vector<int64_t> out_lens(lens.begin(), lens.end());
        if(min_arg->get_shape() != s)
        {
            min_arg = info.add_instruction(make_op("multibroadcast", {{"output_lens", out_lens}}),
                                           min_arg);
        }
        if(max_arg->get_shape() != s)
        {
            max_arg = info.add_instruction(make_op("multibroadcast", {{"output_lens", out_lens}}),
                                           max_arg);
        }

        auto saturated = info.add_instruction(make_op("clip"), add_zero_point, min_arg, max_arg);
        return info.add_instruction(make_op("convert", {{"target_type", quant_type}}), saturated);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
