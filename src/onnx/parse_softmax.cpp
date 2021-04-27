#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softmax : op_parser<parse_softmax>
{
    std::vector<op_desc> operators() const
    {
        return {{"Softmax", "softmax"}, {"LogSoftmax", "logsoftmax"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        // default axis value is -1 for opset 13
        int64_t axis = -1;

        // axis value is 1 for previous opset versions
        if(parser.opset_version < 13)
        {
            axis = 1;
        }

        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        return info.add_instruction(make_op(opd.op_name, {{"axis", axis}}), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
