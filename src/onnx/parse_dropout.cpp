#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_dropout : op_parser<parse_dropout>
{
    std::vector<op_desc> operators() const { return {{"Dropout"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& /*parser*/,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        auto out = info.add_instruction(make_op("identity"), args[0]);
        auto s   = args[0]->get_shape();
        std::vector<int8_t> vec(s.elements(), 1);
        shape mask_s{shape::bool_type, s.lens()};
        auto mask = info.add_literal(literal(mask_s, vec));

        return {out, mask};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
