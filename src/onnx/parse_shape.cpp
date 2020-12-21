#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Use a literal instruction to replace the shape since, output of
// shape operator are literals in migraphx
struct parse_shape : op_parser<parse_shape>
{
    std::vector<op_desc> operators() const { return {{"Shape"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.size() != 1)
            MIGRAPHX_THROW("Shape: operator should have 1 operand");
        std::vector<std::size_t> arg_shape = args[0]->get_shape().lens();
        std::vector<int64_t> vec_shape(arg_shape.size());
        migraphx::shape s(migraphx::shape::int64_type, {arg_shape.size()});
        std::transform(arg_shape.begin(), arg_shape.end(), vec_shape.begin(), [](auto i) {
            return int64_t(i);
        });
        return info.add_literal(migraphx::literal{s, vec_shape});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
