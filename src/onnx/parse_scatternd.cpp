#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_scatternd : op_parser<parse_scatternd>
{
    std::vector<op_desc> operators() const { return {{"ScatterND"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref>& args) const
    {
        if(contains(info.attributes, "reduction"))
        {
            if(info.attributes.at("reduction").s() == "add")
                return info.add_instruction(migraphx::make_op("scatternd_add"), args);
            if(info.attributes.at("reduction").s() == "mul")
                return info.add_instruction(migraphx::make_op("scatternd_mul"), args);
        }

        return info.add_instruction(migraphx::make_op("scatternd_none"), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
