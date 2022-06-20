#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_batchnorm : op_parser<parse_batchnorm>
{
    bool transpose() const { return true; }
    std::vector<op_desc> operators() const { return {{"FusedBatchNorm"}, {"FusedBatchNormV3"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        float epsilon  = 1e-5f;
        float momentum = 0.9f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = info.attributes.at("epsilon").f();
        }
        auto op = make_op("batch_norm_inference", {{"epsilon", epsilon}, {"momentum", momentum}});
        return info.add_instruction(op, args);
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
