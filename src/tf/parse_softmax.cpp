#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_softmax : op_parser<parse_softmax>
{
    std::vector<op_desc> operators() const { return {{"Softmax"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        int axis      = -1;
        auto num_dims = args[0]->get_shape().lens().size();
        if(contains(info.attributes, "axis"))
        {
            axis = static_cast<int>(info.attributes.at("axis").i());
        }

        axis = tune_axis(num_dims, axis, "tf_parse_softmax");

        return info.add_instruction(make_op("softmax", {{"axis", axis}}),
                                    info.make_contiguous(args[0]));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
