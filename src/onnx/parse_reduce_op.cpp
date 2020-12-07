#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

    instruction_ref parse_reduce_oper(const std::string& op_name,
                                      const onnx_parser& parser,
                                      onnx_parser::node_info info,
                                      std::vector<instruction_ref> args)
    {
        std::size_t n_dim = args.front()->get_shape().lens().size();

        // default to reduce over all dimensions
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
            return info.add_instruction(make_op(op_name, {{"axes", axes}}), std::move(args));
        }
        else
        {
            auto ins = info.add_instruction(make_op(op_name, {{"axes", axes}}), std::move(args));
            return info.add_instruction(make_op("squeeze", {{"axes", axes}}), ins);
        }
    }

struct parse_reduce_op : op_parser<parse_reduce_op>
{
    std::vector<op_desc> operators() const { return {
      {"ReduceMax", "reduce_max"},
      {"ReduceMean", "reduce_mean"},
      {"ReduceMin", "reduce_min"},
      {"ReduceProd", "reduce_prod"},
      {"ReduceSum", "reduce_sum"}
    }; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return parse_reduce_oper(opd.op_name, parser, std::move(info), std::move(args));
    }
};

struct parse_reduce_l1 : op_parser<parse_reduce_l1>
{
    std::vector<op_desc> operators() const { return {{"ReduceL1"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto abs_ins = info.add_instruction(make_op("abs"), args[0]);
        return parse_reduce_oper("reduce_sum", parser, std::move(info), {abs_ins});
    }
};

struct parse_reduce_l2 : op_parser<parse_reduce_l2>
{
    std::vector<op_desc> operators() const { return {{"ReduceL2"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto square_ins = info.add_instruction(make_op("mul"), args[0], args[0]);
        auto sum_ins    = parse_reduce_oper("reduce_sum", parser, std::move(info), {square_ins});
        return info.add_instruction(make_op("sqrt"), sum_ins);
    }
};

struct parse_reduce_log_sum : op_parser<parse_reduce_log_sum>
{
    std::vector<op_desc> operators() const { return {{"ReduceLogSum"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto sum_ins = parse_reduce_oper("reduce_sum", parser, std::move(info), std::move(args));
        return info.add_instruction(make_op("log"), sum_ins);
    }
};

struct parse_reduce_log_sum_exp : op_parser<parse_reduce_log_sum_exp>
{
    std::vector<op_desc> operators() const { return {{"ReduceLogSumExp"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto exp_ins = info.add_instruction(make_op("exp"), args[0]);
        auto sum_ins = parse_reduce_oper("reduce_sum", parser, std::move(info), {exp_ins});
        return info.add_instruction(make_op("log"), sum_ins);
    }
};

struct parse_reduce_sum_square : op_parser<parse_reduce_sum_square>
{
    std::vector<op_desc> operators() const { return {{"ReduceSumSquare"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto square_ins = info.add_instruction(make_op("mul"), args[0], args[0]);
        return parse_reduce_oper("reduce_sum", parser, std::move(info), {square_ins});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
