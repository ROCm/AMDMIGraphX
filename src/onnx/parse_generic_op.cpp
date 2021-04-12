#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_generic_op : op_parser<parse_generic_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"Abs", "abs"},
                {"Acos", "acos"},
                {"Acosh", "acosh"},
                {"Asin", "asin"},
                {"Asinh", "asinh"},
                {"Atan", "atan"},
                {"Atanh", "atanh"},
                {"Ceil", "ceil"},
                {"Concat", "concat"},
                {"Cos", "cos"},
                {"Cosh", "cosh"},
                {"Elu", "elu"},
                {"Erf", "erf"},
                {"Exp", "exp"},
                {"Flatten", "flatten"},
                {"Floor", "floor"},
                {"Gather", "gather"},
                {"Identity", "identity"},
                {"LeakyRelu", "leaky_relu"},
                {"Log", "log"},
                {"LRN", "lrn"},
                {"Neg", "neg"},
                {"Reciprocal", "recip"},
                {"Relu", "relu"},
                {"Round", "round"},
                {"Sigmoid", "sigmoid"},
                {"Sign", "sign"},
                {"Sin", "sin"},
                {"Sinh", "sinh"},
                {"Sqrt", "sqrt"},
                {"Tan", "tan"},
                {"Tanh", "tanh"},
                {"Not", "not"}};
    }

    bool needs_contiguous(const std::string& op_name) const
    {
        return contains({"gather"}, op_name);
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto op = parser.load(opd.op_name, info);
        if(needs_contiguous(opd.op_name))
        {
            std::transform(args.begin(), args.end(), args.begin(), [&](auto arg) {
                return info.make_contiguous(arg);
            });
        }
        return info.add_instruction(op, args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
