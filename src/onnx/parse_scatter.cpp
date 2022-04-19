#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scatter : op_parser<parse_scatter>
{
    std::vector<op_desc> operators() const { return {{"ScatterElements"}, {"Scatter"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        operation op;

        std::string op_name = "scatter_none";
        int axis            = 0;

        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();
        if(contains(info.attributes, "reduction"))
        {
            std::string reduction_att(info.attributes.at("reduction").s());
            // check for a valid reduction attribute.  We have an operator for each one.
            if(not contains({"none", "add", "mul"}, reduction_att))
                MIGRAPHX_THROW("PARSE_SCATTER: unsupported reduction mode " + reduction_att);
            // merge scatter with reduction attribute to specify which scatter operation.  Future
            // reduction op names should follow this pattern and should also be added to the check
            // above.
            op_name = std::string("scatter_") + reduction_att;
        }
        op = migraphx::make_op(op_name, {{"axis", axis}});
        return info.add_instruction(op, args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
