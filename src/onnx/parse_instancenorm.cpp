#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_instancenorm : op_parser<parse_instancenorm>
{
    std::vector<op_desc> operators() const { return {{"InstanceNormalization"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        // y = scale * ( x - mean ) / sqrt ( variance + epsilon ) + bias
        // mean = reduce_mean({D1, D2, ... Dk}, x)
        // variance = reduce_mean({D1, D2, ... Dk}, (x - mean)^2)

        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        auto x     = args[0];
        auto scale = args[1];
        auto bias  = args[2];
        auto dims  = x->get_shape().lens();
        auto ndims = dims.size();
        assert(ndims >= 2);
        auto kdims = ndims - 2;

        std::vector<int64_t> axes(kdims);
        std::iota(axes.begin(), axes.end(), 2);

        auto mean = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), x);
        auto mean_bcast =
            info.add_instruction(make_op("multibroadcast", {{"output_lens", dims}}), mean);
        auto l0              = info.add_instruction(make_op("sqdiff"), x, mean_bcast);
        auto variance        = info.add_instruction(make_op("reduce_mean", {{"axes", axes}}), l0);
        auto l1              = info.add_instruction(make_op("sub"), x, mean_bcast);
        auto epsilon_literal = info.add_literal(epsilon);
        auto epsilon_bcast   = info.add_instruction(
            make_op("multibroadcast", {{"output_lens", dims}}), epsilon_literal);
        auto variance_bcast =
            info.add_instruction(make_op("multibroadcast", {{"output_lens", dims}}), variance);
        auto l2 = info.add_instruction(make_op("add"), variance_bcast, epsilon_bcast);
        auto l3 = info.add_instruction(make_op("rsqrt"), l2);
        auto l4 = info.add_instruction(make_op("mul"), l1, l3);
        auto scale_bcast =
            info.add_instruction(make_op("broadcast", {{"axis", 1}, {"dims", dims}}), scale);
        ;
        auto bias_bcast =
            info.add_instruction(make_op("broadcast", {{"axis", 1}, {"dims", dims}}), bias);
        auto l5 = info.add_instruction(make_op("mul"), l4, scale_bcast);
        return info.add_instruction(make_op("add"), l5, bias_bcast);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
