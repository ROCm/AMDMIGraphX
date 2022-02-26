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
        // // The ScatterElements op replaces Scatter (deprecated in Onnx) and has three
        // // possible values for the attribute "reduction", which we implement as
        // // different structs.
        // operation op;

        // int axis = 0;
        // std::string name_of_op("scatter_none");

        // if(contains(info.attributes, "axis"))
        // {
        //     axis = info.attributes.at("axis").i();
        // }
        // if(contains(info.attributes, "reduction"))
        // {
        //     // valid possibilities for Onnx ScatterElements are {add, mul, none}
        //     name_of_op ="scatter_" + info.attributes.at("reduction").s();
        // }

        // // should throw an error if invalid reduction string is given            
        // op = migraphx::make_op(name_of_op, {{"axis", axis}});

        // return info.add_instruction(op, args);
        // The ScatterElements op replaces Scatter (deprecated in Onnx) and has three
        // possible values for the attribute "reduction", which we implement as
        // different structs.
        operation op;

        int axis = 0;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();
        if(contains(info.attributes, "reduction"))
        {
            if(info.attributes.at("reduction").s() == "add")
                op = migraphx::make_op("scatter_add", {{"axis", axis}});
            // return info.add_instruction(migraphx::make_op("scatter_add"), args);
            else if(info.attributes.at("reduction").s() == "mul")
                op = migraphx::make_op("scatter_mul", {{"axis", axis}});
            // return info.add_instruction(migraphx::make_op("scatter_mul"), args);
            else
                op = migraphx::make_op("scatter_none", {{"axis", axis}});
        }
        else
            op = migraphx::make_op("scatter_none", {{"axis", axis}});
        // return info.add_instruction(migraphx::make_op("scatter_none"), args);

        return info.add_instruction(op, args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
