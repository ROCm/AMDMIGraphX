#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_expand : op_parser<parse_expand>
{
    std::vector<op_desc> operators() const { return {{"Expand"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto in_lens             = args[0]->get_shape().lens();
        migraphx::argument arg_s = args[1]->eval();
        check_arg_empty(arg_s, "Expand: dynamic shape is not supported");
        std::vector<std::size_t> dims;
        arg_s.visit([&](auto input) { dims.assign(input.begin(), input.end()); });
        auto out_lens = compute_broadcasted_lens(in_lens, dims);
        return info.add_instruction(make_op("multibroadcast", {{"output_lens", out_lens}}),
                                    args[0]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
