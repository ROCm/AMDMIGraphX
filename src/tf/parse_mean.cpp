#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_mean : op_parser<parse_mean>
{
    std::vector<op_desc> operators() const { return {{"Mean"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        bool keep_dims = info.attributes.at("keep_dims").b();
        auto axes      = args[1]->eval().get<int32_t>().to_vector<int64_t>();

        auto ins = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), args[0]);
        if(not keep_dims)
            ins = info.add_instruction(make_op("squeeze", {{"axes", axes}}), ins);
        return ins;
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
