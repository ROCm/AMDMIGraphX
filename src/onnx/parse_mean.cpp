#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_mean : op_parser<parse_mean>
{
    std::vector<op_desc> operators() const { return {{"Mean"}}; }

    /// Calculates the element-wise mean of n>=1 input tensors
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto num_data = args.size();
        if(num_data == 1)
            return args[0];

        auto divisor = info.add_literal(
            migraphx::literal{migraphx::shape{args[0]->get_shape().type()}, {num_data}});

        return std::accumulate(args.begin(), args.end(), args[0], [&](auto& mean, auto& data_i) {
            // Pre-divide each tensor element-wise by n to reduce risk of overflow during summation
            data_i = info.add_broadcastable_binary_op("div", data_i, divisor);

            if(data_i != args[0])
                return info.add_broadcastable_binary_op("add", mean, data_i);
            return data_i;
        });
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
