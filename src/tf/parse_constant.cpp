#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_constant_op : op_parser<parse_constant_op>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"Const"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& parser,
                          tf_parser::node_info info,
                          const std::vector<instruction_ref>& /*args*/) const
    {
        literal v = parser.parse_tensor(info.attributes.at("value").tensor());
        return info.add_literal(v);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
