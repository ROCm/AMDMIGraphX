#include <migraphx/tf/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_biasadd : op_parser<parse_biasadd>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"BiasAdd"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        uint64_t axis = 1; // assume output of previous layer is in NCHW (broadcast on channel)

        auto l0 = info.add_instruction(
            make_op("broadcast", {{"axis", axis}, {"out_lens", args[0]->get_shape().lens()}}),
            args[1]);
        return info.add_instruction(make_op("add"), args[0], l0);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
