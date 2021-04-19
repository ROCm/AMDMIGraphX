#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_squeeze : op_parser<parse_squeeze>
{
    std::vector<op_desc> operators() const
    {
        return {{"Squeeze", "squeeze"}, {"Unsqueeze", "unsqueeze"}};
    }

    operation assign_axes(operation& op, const std::vector<int64_t>& axes) const
    {
        auto v    = op.to_value();
        v["axes"] = axes;
        op.from_value(v);

        return op;
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto op = parser.load(opd.op_name, info);
        std::vector<int64_t> axes;
        if(args.size() == 2)
        {
            auto arg_axes = args.at(1)->eval();
            check_arg_empty(arg_axes, "PARSE_" + opd.op_name + ": cannot handle variable axes!");
            arg_axes.visit([&](auto s) { axes.assign(s.begin(), s.end()); });
            op = assign_axes(op, axes);
        }

        auto arg = info.make_contiguous(args.front());
        return info.add_instruction(op, arg);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
