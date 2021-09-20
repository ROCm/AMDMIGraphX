#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_clip : op_parser<parse_clip>
{
    std::vector<op_desc> operators() const { return {{"Clip"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto input_lens = args[0]->get_shape().lens();
        instruction_ref min_arg;
        instruction_ref max_arg;
        bool min_used = false;
        bool max_used = false;

        if(args.size() == 3 and args[2]->name() != "undefined")
        {
            max_arg  = args[2];
            max_used = true;
        }

        if(args.size() >= 2 and args[1]->name() != "undefined")
        {
            min_arg  = args[1];
            min_used = true;
        }
        // if using previous opset for attributes
        else if(contains(info.attributes, "min") and contains(info.attributes, "max"))
        {

            float min_val = parser.parse_value(info.attributes.at("min")).at<float>();
            float max_val = parser.parse_value(info.attributes.at("max")).at<float>();
            min_arg       = info.add_literal(min_val);
            max_arg       = info.add_literal(max_val);
            min_used      = true;
            max_used      = true;
        }

        if(min_used)
        {
            min_arg = info.add_instruction(make_op("multibroadcast", {{"out_lens", input_lens}}),
                                           min_arg);
        }

        if(max_used)
        {
            max_arg = info.add_instruction(make_op("multibroadcast", {{"out_lens", input_lens}}),
                                           max_arg);
        }

        if(min_used and max_used)
        {
            return info.add_instruction(make_op("clip"), args[0], min_arg, max_arg);
        }
        else if(max_used)
        {
            return info.add_instruction(make_op("min"), args[0], max_arg);
        }
        else if(min_used)
        {
            return info.add_instruction(make_op("max"), args[0], min_arg);
        }
        else
        {
            return info.add_instruction(make_op("identity"), args[0]);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
