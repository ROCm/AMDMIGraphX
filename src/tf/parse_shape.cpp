#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_shape : op_parser<parse_shape>
{
    std::vector<op_desc> operators() const { return {{"Shape"}}; }

    // Use a literal instruction to replace the shape since output of
    // shape operator are literals in migraphx
    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<std::size_t> arg_shape = args[0]->get_shape().lens();
        std::vector<int32_t> vec_shape(arg_shape.size());
        migraphx::shape s(migraphx::shape::int32_type, {arg_shape.size()});
        std::transform(
            arg_shape.begin(), arg_shape.end(), vec_shape.begin(), [](auto i) { return i; });
        return info.add_literal(migraphx::literal{s, vec_shape});
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
