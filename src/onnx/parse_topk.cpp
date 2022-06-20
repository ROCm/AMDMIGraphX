#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_topk : op_parser<parse_topk>
{
    std::vector<op_desc> operators() const { return {{"TopK"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        int64_t k = 0;
        if(args.size() == 2)
        {
            auto arg_k = args.at(1)->eval();
            check_arg_empty(arg_k, "PARSE_TopK: k input must be constant");
            k = arg_k.at<int>();
        }
        else if(contains(info.attributes, "k"))
        {
            k = info.attributes.at("k").i();
        }

        bool largest = true;
        if(contains(info.attributes, "largest"))
        {
            largest = static_cast<bool>(info.attributes.at("largest").i());
        }

        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        auto topk_ret = info.add_instruction(
            make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0));

        auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
        auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);

        return {ret_val, ret_ind};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
