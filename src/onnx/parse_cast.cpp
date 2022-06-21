#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_cast : op_parser<parse_cast>
{
    std::vector<op_desc> operators() const { return {{"Cast"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        if(!contains(info.attributes, "to"))
        {
            MIGRAPHX_THROW("PARSE_CAST: missing to type attribute!");
        }

        int to_type        = parser.parse_value(info.attributes.at("to")).at<int>();
        shape::type_t type = get_type(to_type);
        return info.add_instruction(make_op("convert", {{"target_type", type}}), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
