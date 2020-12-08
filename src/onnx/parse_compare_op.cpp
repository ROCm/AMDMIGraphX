#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_compare_op : op_parser<parse_compare_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"Equal", "equal"}, {"Greater", "greater"}, {"Less", "less"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto l = info.add_broadcastable_binary_op(opd.op_name, args[0], args[1]);
        if(l->get_shape().type() != shape::bool_type)
        {
            l = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), l);
        }
        return l;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
