#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_relu6 : op_parser<parse_relu6>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"Relu6"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input_lens = args[0]->get_shape().lens();
        auto min_val    = info.add_literal(0.0f);
        auto max_val    = info.add_literal(6.0f);

        min_val =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", input_lens}}), min_val);
        max_val =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", input_lens}}), max_val);
        return info.add_instruction(make_op("clip"), args.front(), min_val, max_val);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
