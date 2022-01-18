#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softplus : op_parser<parse_softplus>
{
    std::vector<op_desc> operators() const { return {{"Softplus"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // Apply pointwise formula: y = ln(exp(x) + 1)
        auto mb_ones = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", args[0]->get_shape().lens()}}),
            info.add_literal(migraphx::literal{migraphx::shape{args[0]->get_shape().type()}, {1}}));
        auto exp = info.add_instruction(migraphx::make_op("exp"), args[0]);
        auto add = info.add_instruction(migraphx::make_op("add"), exp, mb_ones);
        return info.add_instruction(migraphx::make_op("log"), add);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
