#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/batch_norm_inference.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_batchnorm : op_parser<parse_batchnorm>
{
    std::vector<op_desc> operators() const { return {{"BatchNormalization"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        float epsilon                                     = 1e-5f;
        float momentum                                    = 0.9f;
        op::batch_norm_inference::bn_infer_mode_t bn_mode = op::batch_norm_inference::spatial;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        if(contains(info.attributes, "momentum"))
        {
            momentum = parser.parse_value(info.attributes.at("momentum")).at<float>();
        }
        if(contains(info.attributes, "spatial"))
        {
            bn_mode = (parser.parse_value(info.attributes.at("spatial")).at<uint64_t>() > 0)
                          ? op::batch_norm_inference::spatial
                          : op::batch_norm_inference::per_activation;
        }
        op::batch_norm_inference op{epsilon, momentum, bn_mode};
        return info.add_instruction(op, args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
