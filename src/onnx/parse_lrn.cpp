#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_lrn : op_parser<parse_lrn>
{
    std::vector<op_desc> operators() const { return {{"LRN", "lrn"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto op   = parser.load(opd.op_name, info);
        auto& arg = args.front();
        auto type = arg->get_shape().type();
        if(type == shape::half_type)
        {
            arg =
                info.add_instruction(make_op("convert", {{"target_type", shape::float_type}}), arg);
        }
        auto ret = info.add_instruction(op, arg);
        if(type == shape::half_type)
        {
            ret =
                info.add_instruction(make_op("convert", {{"target_type", shape::half_type}}), ret);
        }
        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
