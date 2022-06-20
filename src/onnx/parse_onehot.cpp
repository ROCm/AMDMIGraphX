#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_onehot : op_parser<parse_onehot>
{
    std::vector<op_desc> operators() const { return {{"OneHot"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        migraphx::argument depth_arg = args[1]->eval();
        check_arg_empty(depth_arg, "PARSE_ONEHOT: depth - dynamic shape not supported");
        size_t depth = depth_arg.at<size_t>();

        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
        }

        std::vector<float> depth_input(depth * depth, 0.0f);
        for(int i = 0; i < depth; i++)
        {
            depth_input[depth * i + i] = 1.0f;
        }

        auto type = args[2]->get_shape().type();
        shape s{type, {depth, depth}};
        auto l_val      = info.add_literal({s, depth_input});
        auto gather_out = info.add_instruction(make_op("gather", {{"axis", 0}}), {l_val, args[0]});

        // Finally, we need a transpose to move the inner most dim to the axis dim
        int n_rank         = gather_out->get_shape().lens().size();
        int64_t tuned_axis = tune_axis(n_rank, axis, opd.op_name);
        std::vector<int64_t> perm(n_rank - 1);
        std::iota(perm.begin(), perm.end(), 0);
        perm.insert(perm.begin() + tuned_axis, n_rank - 1);
        auto tr_out =
            info.add_instruction(make_op("transpose", {{"permutation", perm}}), gather_out);
        auto lens = tr_out->get_shape().lens();

        auto off_val = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), args[2]);
        auto on_val = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), args[2]);
        auto diff = info.add_instruction(make_op("sub"), on_val, off_val);
        auto unsq_off_val =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), off_val);
        auto unsq_diff_val =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}), diff);
        auto l_mul = info.add_instruction(make_op("mul"), tr_out, unsq_diff_val);
        return info.add_instruction(make_op("add"), l_mul, unsq_off_val);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
