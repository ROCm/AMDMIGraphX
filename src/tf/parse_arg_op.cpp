#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_arg_op : op_parser<parse_arg_op>
{
    std::vector<op_desc> operators() const { return {{"ArgMax", "argmax"}, {"ArgMin", "argmin"}}; }

    instruction_ref parse(const op_desc& opd,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        int64_t axis = 0;
        axis         = args[1]->eval().at<int64_t>();
        auto ins     = info.add_instruction(make_op(opd.op_name, {{"axis", axis}}), args.front());
        return info.add_instruction(make_op("squeeze", {{"axes", {axis}}}), ins);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
