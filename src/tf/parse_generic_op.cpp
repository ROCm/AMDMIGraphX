#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_generic_op : op_parser<parse_generic_op>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const
    {
        return {{"All", "identity"},
                {"Identity", "identity"},
                {"LessEqual", "identity"},
                {"Relu", "relu"},
                {"Rsqrt", "rsqrt"},
                {"Tanh", "tanh"},
                {"StopGradient", "identity"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const tf_parser& /*parser*/,
                          const tf_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        return info.add_instruction(make_op(opd.op_name), args);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
