#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_binary_op : op_parser<parse_binary_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"Add", "add"},
                {"Div", "div"},
                {"And", "logical_and"},
                {"Or", "logical_or"},
                {"Xor", "logical_xor"},
                {"Mul", "mul"},
                {"PRelu", "prelu"},
                {"Sub", "sub"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 2)
            MIGRAPHX_THROW("binary operators should have 2 operands");
        if(contains(info.attributes, "broadcast") and contains(info.attributes, "axis"))
        {
            uint64_t broadcasted =
                parser.parse_value(info.attributes.at("broadcast")).at<uint64_t>();
            if(broadcasted != 0)
            {
                uint64_t axis = parser.parse_value(info.attributes.at("axis")).at<uint64_t>();
                auto l        = info.add_instruction(
                    make_op("broadcast",
                            {{"axis", axis}, {"out_lens", args[0]->get_shape().lens()}}),
                    args[1]);
                return info.add_instruction(make_op(opd.op_name), args[0], l);
            }
            return info.add_instruction(make_op(opd.op_name), args);
        }
        else
        {
            return info.add_broadcastable_binary_op(opd.op_name, args[0], args[1]);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
