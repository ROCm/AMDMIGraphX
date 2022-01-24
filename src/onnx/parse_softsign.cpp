#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softsign : op_parser<parse_softsign>
{
    std::vector<op_desc> operators() const { return {{"Softsign"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // Apply pointwise formula: y = x / (1 + |x|)
        auto mb_ones = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", args[0]->get_shape().lens()}}),
            info.add_literal(migraphx::literal{migraphx::shape{args[0]->get_shape().type()}, {1}}));
        auto abs = info.add_instruction(migraphx::make_op("abs"), args[0]);
        auto add = info.add_instruction(migraphx::make_op("add"), abs, mb_ones);
        return info.add_instruction(migraphx::make_op("div"), args[0], add);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
