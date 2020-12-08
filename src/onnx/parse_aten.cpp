#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

enum class reduce_mode_t
{
    sum  = 0,
    mean = 1,
    max  = 2
};

struct parse_aten : op_parser<parse_aten>
{
    std::vector<op_desc> operators() const { return {{"ATen"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(contains(info.attributes, "operator"))
        {
            auto op_name = info.attributes.at("operator").s();
            if(op_name.find("embedding_bag") != std::string::npos)
            {
                return parse_embedding_bag(info, std::move(args));
            }
        }
        MIGRAPHX_THROW("PARSE_ATEN: unsupported custom operator");
    }

    instruction_ref parse_embedding_bag(onnx_parser::node_info info,
                                        std::vector<instruction_ref> args) const
    {
        if(args[2]->get_shape().elements() != 1)
            MIGRAPHX_THROW("PARSE_EMBEDDING_BAG: MIGraphX only supports offsets of size 1");
        reduce_mode_t reduce_mode = reduce_mode_t::sum;
        if(contains(info.attributes, "mode"))
        {
            reduce_mode = static_cast<reduce_mode_t>(info.attributes.at("mode").i());
        }

        auto l0 = info.add_instruction(make_op("gather"), args[0], args[1]);
        switch(reduce_mode)
        {
        case reduce_mode_t::sum:
            l0 = info.add_instruction(make_op("reduce_sum", {{"axes", {0}}}), l0);
            break;
        case reduce_mode_t::mean:
            l0 = info.add_instruction(make_op("reduce_mean", {{"axes", {0}}}), l0);
            break;
        case reduce_mode_t::max:
            l0 = info.add_instruction(make_op("reduce_max", {{"axes", {0}}}), l0);
            break;
        }
        return l0;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
