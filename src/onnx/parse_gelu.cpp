#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gelu : op_parser<parse_gelu>
{
    std::vector<op_desc> operators() const { return {{"Gelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /* parser */,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("Gelu: too many arguments. Expected 1; got " +
                           std::to_string(args.size()));

        auto x        = args.front();
        auto x_type   = x->get_shape().type();
        auto root_inv = info.add_literal(literal{shape{x_type, {1}}, {1.0f / std::sqrt(2.0f)}});
        auto product  = info.add_broadcastable_binary_op("mul", x, root_inv);
        auto erf      = info.add_instruction(make_op("erf"), product);
        auto one      = info.add_literal(literal{shape{x_type, {1}}, {1.0f}});
        erf           = info.add_broadcastable_binary_op("add", one, erf);
        auto half     = info.add_literal(literal{shape{x_type, {1}}, {0.5f}});
        erf           = info.add_broadcastable_binary_op("mul", half, erf);

        return info.add_instruction(make_op("mul"), x, erf);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
