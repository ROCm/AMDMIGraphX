#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_variadic_op : op_parser<parse_variadic_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"Sum", "add"}, {"Max", "max"}, {"Min", "min"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser&,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return std::accumulate(std::next(args.begin()),
                               args.end(),
                               args.front(),
                               [&](instruction_ref a, instruction_ref b) {
                                   return info.add_broadcastable_binary_op(opd.op_name, a, b);
                               });
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
