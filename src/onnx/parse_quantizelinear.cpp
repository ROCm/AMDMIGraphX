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
                          const std::vector<instruction_ref>& args) const
    {
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        return info.add_instruction(make_op("quantizelinear", {{"axis", axis}}), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
