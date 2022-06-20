#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_cast : op_parser<parse_cast>
{
    std::vector<op_desc> operators() const { return {{"Cast"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        shape::type_t type = parser.parse_type(info.attributes.at("DstT").type());
        return info.add_instruction(make_op("convert", {{"target_type", type}}), args);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
