#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/layernorm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_layernorm : op_parser<parse_layernorm>
{
    std::vector<op_desc> operators() const { return {{"LayerNormalization"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        float epsilon = 1e-3f;
        int64_t axis  = -1;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        if(contains(info.attributes, "axis"))
        {
            epsilon = parser.parse_value(info.attributes.at("axis")).at<int64_t>();
        }

        auto layernorm = info.add_instruction(
            make_op("layernorm", {{"epsilon", epsilon}, {"axis", axis}}), args.front());

        if(args.size() >= 2)
            layernorm = info.add_instruction(make_op("mul"), layernorm, args.at(1));
        if(args.size() == 3)
            layernorm = info.add_instruction(make_op("add"), layernorm, args.at(2));

        return layernorm;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
