#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_dequantizelinear : op_parser<parse_dequantizelinear>
{
    std::vector<op_desc> operators() const { return {{"DequantizeLinear"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto input_lens = args[0]->get_shape().lens();
        auto n_dim      = input_lens.size();

        instruction_ref x_scale;
        if(args[1]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, opd.op_name);
            x_scale         = info.add_instruction(
                make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), args[1]);
        }
        else
        {
            x_scale = info.add_instruction(make_op("multibroadcast", {{"output_lens", input_lens}}),
                                           args[1]);
        }

        if(args.size() == 3)
        {
            auto x_zero_point = args[2];
            if(x_zero_point->get_shape().elements() != 1)
            {
                auto tuned_axis = tune_axis(n_dim, axis, opd.op_name);
                x_zero_point    = info.add_instruction(
                    make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}),
                    x_zero_point);
            }
            else
            {
                x_zero_point = info.add_instruction(
                    make_op("multibroadcast", {{"output_lens", input_lens}}), x_zero_point);
            }

            return info.add_instruction(
                make_op("dequantizelinear"), args[0], x_scale, x_zero_point);
        }

        return info.add_instruction(make_op("dequantizelinear"), args[0], x_scale);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
