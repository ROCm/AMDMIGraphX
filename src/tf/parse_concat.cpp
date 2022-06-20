#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_concat : op_parser<parse_concat>
{
    std::vector<op_desc> operators() const { return {{"ConcatV2"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // get index for axis within args
        size_t axis_idx = info.attributes.at("N").i();
        int64_t axis    = args[axis_idx]->eval().at<int64_t>();
        auto op         = make_op("concat", {{"axis", axis}});
        // return only first N arguments (assuming last index is the axis value)
        return info.add_instruction(
            op, std::vector<instruction_ref>(args.begin(), args.begin() + args.size() - 1));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
