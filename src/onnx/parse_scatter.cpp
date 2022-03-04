#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scatter : op_parser<parse_scatter>
{
    std::vector<op_desc> operators() const { return {{"ScatterElements"}}; }

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
            if(info.attributes.at("reduction").s() == "add")
                op_name = "scatter_add";
            else if(info.attributes.at("reduction").s() == "mul")
                op_name = "scatter_mul";
            else
                op_name = "scatter_none";
        }

        op = migraphx::make_op(op_name, {{"axis", axis}});
        return info.add_instruction(op, args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
