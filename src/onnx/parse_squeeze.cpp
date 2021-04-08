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

    bool needs_contiguous(const std::string& op_name) const
    {
        return contains({"squeeze", "unsqueeze"}, op_name);
    }

    void assign_axes(operation& op, const std::vector<int64_t>& axes) const
    {
        auto v = op.to_value();
        for(auto&& x : v)
        {
            if(x.get_key() == "axes")
            {
                x = axes;
                break;
            }
        }
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto op = parser.load(opd.op_name, info);
        std::vector<int64_t> axes;
        if(args.size() == 2)
        {
            auto arg_axes = args.at(1)->eval();
            check_arg_empty(arg_axes, "PARSE_" + opd.op_name + ": cannot handle variable axes!");
            arg_axes.visit([&](auto s) { axes.assign(s.begin(), s.end()); });
            assign_axes(op, axes);
        }
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
