#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_binary_op : op_parser<parse_binary_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"Add", "add"},
                {"AddV2", "add"},
                {"Mul", "mul"},
                {"Pow", "pow"},
                {"SquaredDifference", "sqdiff"},
                {"Sub", "sub"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 2)
            MIGRAPHX_THROW("binary operators should have 2 operands");
        return info.add_broadcastable_binary_op(opd.op_name, args[0], args[1]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
