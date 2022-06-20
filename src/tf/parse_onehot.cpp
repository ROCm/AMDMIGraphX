#include <migraphx/tf/op_parser.hpp>
#include <migraphx/tf/tf_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace tf {

struct parse_onehot : op_parser<parse_onehot>
{
    std::vector<op_desc> operators() const { return {{"OneHot"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const tf_parser& /*parser*/,
                          tf_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        size_t depth = static_cast<size_t>(args[1]->eval().at<int32_t>());

        int64_t axis    = -1;
        float on_value  = args[2]->eval().at<float>();
        float off_value = args[3]->eval().at<float>();

        std::vector<float> depth_input(depth * depth, off_value);
        for(int i = 0; i < depth; i++)
        {
            depth_input[depth * i + i] = on_value;
        }

        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();
        if(axis == -1)
        {
            shape s{shape::float_type, {depth, depth}};
            auto l0 = info.add_literal({s, depth_input});
            return info.add_instruction(make_op("gather", {{"axis", 0}}), {l0, args[0]});
        }
        MIGRAPHX_THROW("MIGraphX does not support axis != -1");
    }
};

} // namespace tf
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
