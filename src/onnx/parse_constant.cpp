#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_constant : op_parser<parse_constant>
{
    std::vector<op_desc> operators() const { return {{"Constant"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& /*args*/) const
    {
        literal v = parser.parse_value(info.attributes.at("value"));
        // return empty literal
        if(v.get_shape().elements() == 0)
        {
            return info.add_literal(literal{});
        }

        auto dim_size = info.attributes.at("value").t().dims_size();
        // if dim_size is 0, it is a scalar
        if(dim_size == 0)
        {
            migraphx::shape scalar_shape{v.get_shape().type()};
            return info.add_literal(migraphx::literal{scalar_shape, v.data()});
        }

        return info.add_literal(v);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
