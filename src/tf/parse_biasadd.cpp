#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_biasadd : op_parser<parse_biasadd>
{
    std::vector<op_desc> operators() const { return {{"BiasAdd"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        args          = parser.to_nchw(args);
        uint64_t axis = 1; // assume output of previous layer is in NCHW (broadcast on channel)

        auto l0 = info.add_instruction(
            make_op("broadcast", {{"axis", axis}, {"dims", args[0]->get_shape().lens()}}), args[1]);
        return parser.to_nhwc(info.add_instruction(make_op("add"), args[0], l0));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
