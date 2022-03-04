#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_size : op_parser<parse_size>
{
    std::vector<op_desc> operators() const { return {{"Size"}}; }

    instruction_ref parse(const op_desc&,
                          const onnx_parser&,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto input_lens = args[0]->get_shape().lens();
        int64_t size    = std::accumulate(
            input_lens.cbegin(), input_lens.cend(), int64_t{1}, std::multiplies<>());
        return info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {size}});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
