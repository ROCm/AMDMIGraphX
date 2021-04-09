#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_lessorequal : op_parser<parse_lessorequal>
{
    std::vector<op_desc> operators() const { return {{"LessOrEqual"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto in_res = info.add_broadcastable_binary_op("greater", args[0], args[1]);
        if(in_res->get_shape().type() != shape::bool_type)
        {
            in_res = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                          in_res);
        }
        return info.add_instruction(make_op("not"), in_res);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
