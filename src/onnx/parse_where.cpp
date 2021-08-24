#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_where : op_parser<parse_where>
{
    std::vector<op_desc> operators() const { return {{"Where"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto lens =
            compute_broadcasted_lens(args[0]->get_shape().lens(), args[1]->get_shape().lens());
        lens = compute_broadcasted_lens(lens, args[2]->get_shape().lens());
        if(args[0]->get_shape().lens() != lens)
        {
            args[0] =
                info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), args[0]);
        }

        if(args[1]->get_shape().lens() != lens)
        {
            args[1] =
                info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), args[1]);
        }

        if(args[2]->get_shape().lens() != lens)
        {
            args[2] =
                info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), args[2]);
        }

        return info.add_instruction(make_op("where"), args[0], args[1], args[2]);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
