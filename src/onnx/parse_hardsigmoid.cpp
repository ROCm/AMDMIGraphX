#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_hardsigmoid : op_parser<parse_hardsigmoid>
{
    std::vector<op_desc> operators() const { return {{"HardSigmoid"}, {"HardSwish"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        float alpha = 0.2;
        float beta  = 0.5;
        if(opd.onnx_name == "HardSwish")
        {
            alpha = 1.0 / 6.0;
        }
        else
        {
            if(contains(info.attributes, "alpha"))
                alpha = info.attributes.at("alpha").f();

            if(contains(info.attributes, "beta"))
                beta = info.attributes.at("beta").f();
        }

        auto input_lens = args[0]->get_shape().lens();
        auto input_type = args[0]->get_shape().type();
        auto mb_alpha   = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {alpha}}));
        auto mb_beta = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {beta}}));
        auto mb_zero = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {0}}));
        auto mb_one = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {1}}));

        auto mul         = info.add_instruction(migraphx::make_op("mul"), mb_alpha, args[0]);
        auto add         = info.add_instruction(migraphx::make_op("add"), mb_beta, mul);
        auto hardsigmoid = info.add_instruction(migraphx::make_op("clip"), add, mb_zero, mb_one);
        if(opd.onnx_name == "HardSwish")
            return info.add_instruction(migraphx::make_op("mul"), args[0], hardsigmoid);

        return hardsigmoid;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
