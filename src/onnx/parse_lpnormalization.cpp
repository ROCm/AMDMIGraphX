#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

//!  Parser for LpNormalization ONNX operator.
/*!
  Normalizes a tensor by the L1 or L2 norms along a given axis.
  Norms that evaluate to 0 are changed to 1 to prevent division by zero.
*/
struct parse_lpnormalization : op_parser<parse_lpnormalization>
{
    std::vector<op_desc> operators() const { return {{"LpNormalization"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int p = 2;
        if(contains(info.attributes, "p"))
        {
            p = info.attributes.at("p").i();
        }
        if(p != 1 and p != 2)
        {
            MIGRAPHX_THROW("LPNORMALIZATION: only L1 and L2 norm supported");
        }
        auto input              = args.front();
        auto input_shape        = input->get_shape();
        const auto& input_lens  = input_shape.lens();
        auto input_type         = input_shape.type();
        std::ptrdiff_t num_axes = input_lens.size();
        std::ptrdiff_t axis     = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
            if(axis < -num_axes or axis >= num_axes)
            {
                // handled in normalize_attributes but throwing here might be clearer
                MIGRAPHX_THROW("LPNORMALIZATION: selected axis out of bounds");
            }
        }
        migraphx::instruction_ref p_val;
        if(p == 1)
        {
            p_val = info.add_instruction(migraphx::make_op("abs"), input);
        }
        else
        {
            p_val = info.add_instruction(migraphx::make_op("mul"), input, input);
        }

        // need to check for zeros from lp norm to prevent division by zero
        // change them to 1 for the element-wise division
        auto norms =
            info.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {axis}}}), p_val);
        if(p == 2)
        {
            norms = info.add_instruction(migraphx::make_op("sqrt"), norms);
        }
        // broadcast back to initial shape, negative axis option doesn't work with unidirectional
        norms = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), norms);
        auto zero_mb = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {0.}}));
        auto one_mb = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
            info.add_literal(migraphx::literal{migraphx::shape{input_type}, {1.}}));
        auto is_zero = info.add_instruction(migraphx::make_op("equal"), norms, zero_mb);
        auto norms_zeros_to_one =
            info.add_instruction(migraphx::make_op("where"), is_zero, one_mb, norms);
        return info.add_instruction(migraphx::make_op("div"), input, norms_zeros_to_one);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
