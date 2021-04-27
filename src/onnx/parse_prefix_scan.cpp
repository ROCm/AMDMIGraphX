#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
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
    migraphx::argument in = args[1]->eval();
    check_arg_empty(in, "PARSE_PREFIX_SCAN: axis - dynamic shape not supported");
    std::vector<std::size_t> axis_in;
    in.visit([&](auto input) { axis_in.assign(input.begin(), input.end()); });
    int64_t axis = axis_in[0];

    bool exclusive = false;
    bool reverse   = false;

    if(contains(info.attributes, "exclusive"))
    {
        exclusive = parser.parse_value(info.attributes.at("exclusive")).at<bool>();
    }

    if(contains(info.attributes, "reverse"))
    {
        reverse = parser.parse_value(info.attributes.at("reverse")).at<bool>();
    }

    return info.add_instruction(
        make_op(op_name, {{"axis", axis}, {"exclusive", exclusive}, {"reverse", reverse}}),
        args[0]);
}

struct parse_prefix_scan_op : op_parser<parse_prefix_scan_op>
{
    std::vector<op_desc> operators() const { return {{"CumSum", "prefix_scan_sum"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return parse_prefix_scan_oper(opd.op_name, parser, std::move(info), std::move(args));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
