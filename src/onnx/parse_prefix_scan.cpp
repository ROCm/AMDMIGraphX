#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

instruction_ref parse_prefix_scan_oper(const std::string& op_name,
                                  const onnx_parser& parser,
                                  onnx_parser::node_info info,
                                  std::vector<instruction_ref> args)
{
    std::size_t n_dim = args.front()->get_shape().lens().size();

    std::vector<int64_t> axes(n_dim);
    std::iota(axes.begin(), axes.end(), 0);
    if(contains(info.attributes, "axes"))
    {
        axes.clear();
        auto&& attr_axes = info.attributes["axes"].ints();
        axes             = std::vector<int64_t>(attr_axes.begin(), attr_axes.end());
    }

    int keep_dims = 1;
    if(contains(info.attributes, "keepdims"))
    {
        keep_dims = parser.parse_value(info.attributes.at("keepdims")).at<int>();
    }

    if(keep_dims == 1)
    {
        return info.add_instruction(make_op(op_name, {{"axes", axes}}), args);
    }
    else
    {
        auto ins = info.add_instruction(make_op(op_name, {{"axes", axes}}), args);
        return info.add_instruction(make_op("squeeze", {{"axes", axes}}), ins);
    }
}

struct parse_prefix_scan_op : op_parser<parse_prefix_scan_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"CumSum", "prefix_scan_sum"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return parse_prefix_scan_oper(opd.op_name, parser, std::move(info), std::move(args));
    }
};

struct parse_prefix_scan_l1 : op_parser<parse_prefix_scan_l1>
{
    std::vector<op_desc> operators() const { return {{"ReduceL1"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return parse_prefix_scan_oper("prefix_scan_sum", parser, std::move(info), {args[0]});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
