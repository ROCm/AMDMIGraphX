#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_fastgelu : op_parser<parse_fastgelu>
{
    std::vector<op_desc> operators() const { return {{"FastGelu"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /* parser */,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("FastGelu: too many arguments. Expected 1; got " +
                           std::to_string(args.size()));

        // silu approximation
        auto x       = args.front();
        auto x_type  = x->get_shape().type();
        auto lit     = info.add_literal(literal{shape{x_type, {1}}, {1.702f}});
        auto sigmoid = info.add_broadcastable_binary_op("mul", lit, x);
        sigmoid      = info.add_instruction(make_op("sigmoid"), sigmoid);
        return info.add_instruction(make_op("mul"), sigmoid, x);

        // tanh approximation
        /* auto x = args.front();
        auto x_type = x->get_shape().type();
        auto x3 = info.add_instruction(make_op("mul"), x, x);
        x3 = info.add_instruction(make_op("mul"), x3, x);
        auto magic_number = info.add_literal(literal{shape{x_type, {1}}, {0.044715f}});
        x3 = info.add_broadcastable_binary_op("mul", magic_number, x3);
        auto product = info.add_instruction(make_op("add"), x, x3);
        auto root = info.add_literal(literal{shape{x_type, {1}}, {std::sqrt(2.0 / 3.14159)}});
        product = info.add_broadcastable_binary_op("mul", root, product);
        auto tanh = info.add_instruction(make_op("tanh"), product);
        auto one = info.add_literal(literal{shape{x_type, {1}}, {1.0f}});
        tanh = info.add_broadcastable_binary_op("add", one, tanh);
        auto half = info.add_literal(literal{shape{x_type, {1}}, {0.5f}});
        tanh = info.add_broadcastable_binary_op("mul", half, tanh);

        return info.add_instruction(make_op("mul"), x, tanh); */

        // tanh approximation with pow
        /* auto x = args.front();
        auto x_type = x->get_shape().type();
        auto three = info.add_literal(literal{shape{x_type, {1}}, {3}});
        three = info.add_instruction(make_op("multibroadcast", {{"out_lens",
        x->get_shape().lens()}}), three); auto x3 = info.add_instruction(make_op("pow"), x, three);
        auto magic_number = info.add_literal(literal{shape{x_type, {1}}, {0.044715f}});
        x3 = info.add_broadcastable_binary_op("mul", magic_number, x3);
        auto product = info.add_instruction(make_op("add"), x, x3);
        auto root = info.add_literal(literal{shape{x_type, {1}}, {std::sqrt(2.0 / 3.14159)}});
        product = info.add_broadcastable_binary_op("mul", root, product);
        auto tanh = info.add_instruction(make_op("tanh"), product);
        auto one = info.add_literal(literal{shape{x_type, {1}}, {1.0f}});
        tanh = info.add_broadcastable_binary_op("add", one, tanh);
        auto half = info.add_literal(literal{shape{x_type, {1}}, {0.5f}});
        tanh = info.add_broadcastable_binary_op("mul", half, tanh);

        return info.add_instruction(make_op("mul"), x, tanh); */
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
