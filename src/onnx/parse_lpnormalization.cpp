#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

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
        auto input_shape        = args.front()->get_shape();
        auto input_lens         = input_shape.lens();
        auto input_type         = args[0]->get_shape().type();
        std::ptrdiff_t num_axes = input_lens.size();
        std::ptrdiff_t axis     = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
            if(axis < -num_axes or axis >= num_axes)
            { // handled in normalize_attributes but throwing here might be clearer
                MIGRAPHX_THROW("LPNORMALIZATION: selected axis out of bounds");
            }
        }
        migraphx::instruction_ref p_val;
        if(p == 1)
        {
            p_val = info.add_instruction(migraphx::make_op("abs"), args.front());
        }
        else
        {
            p_val = info.add_instruction(migraphx::make_op("mul"), args.front(), args.front());
        }

        // need to check for zeros from lp norm to prevent division by zero
        // change them to 1 for the element-wise division
        auto norms = info.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axis}}), p_val);
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
        auto is_zero            = info.add_instruction(migraphx::make_op("equal"), norms, zero_mb);
        auto norms_zeros_to_one = info.add_instruction(migraphx::make_op("max"), norms, is_zero);
        return info.add_instruction(migraphx::make_op("div"), args.front(), norms_zeros_to_one);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
