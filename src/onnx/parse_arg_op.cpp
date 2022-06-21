#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_arg_op : op_parser<parse_arg_op>
{
    std::vector<op_desc> operators() const { return {{"ArgMax", "argmax"}, {"ArgMin", "argmin"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = static_cast<int64_t>(parser.parse_value(info.attributes.at("axis")).at<int>());
        }

        int keep_dims = 1;
        if(contains(info.attributes, "keepdims"))
        {
            keep_dims = parser.parse_value(info.attributes.at("keepdims")).at<int>();
        }

        if(keep_dims == 0)
        {
            auto ins = info.add_instruction(make_op(opd.op_name, {{"axis", axis}}), args);
            return info.add_instruction(make_op("squeeze", {{"axes", {axis}}}), ins);
        }
        else
        {
            return info.add_instruction(make_op(opd.op_name, {{"axis", axis}}), args);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
