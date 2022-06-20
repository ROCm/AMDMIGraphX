#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_pack : op_parser<parse_pack>
{
    std::vector<op_desc> operators() const { return {{"Pack"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // reinterpret as unsqueeze with concat
        std::vector<instruction_ref> unsqueezed_args;
        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();
        size_t input_size = args.front()->get_shape().lens().size();
        if(axis > input_size)
        {
            MIGRAPHX_THROW("TF_PARSER: axis value of " + to_string(axis) +
                           " must be smaller than input size " + to_string(input_size));
        }

        std::transform(
            args.begin(),
            args.end(),
            std::back_inserter(unsqueezed_args),
            [&](instruction_ref arg) {
                return info.add_instruction(make_op("unsqueeze", {{"axes", {axis}}}), arg);
            });
        return parser.to_nhwc(
            info.add_instruction(make_op("concat", {{"axis", axis}}), unsqueezed_args));
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
