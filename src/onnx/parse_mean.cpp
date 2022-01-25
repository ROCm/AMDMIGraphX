#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

std::vector<std::size_t> compute_output_lens(std::vector<std::vector<std::size_t>>& input_lens)
{
    // Compute max number of dimensions
    std::vector<std::size_t> lens_sizes;
    std::transform(input_lens.begin(),
                   input_lens.end(),
                   std::back_inserter(lens_sizes),
                   [&](const auto lens) { return lens.size(); });
    auto max_size = *std::max_element(lens_sizes.begin(), lens_sizes.end());

    // Pad each input_lens where size < max_size with 1s
    std::for_each(input_lens.begin(), input_lens.end(), [&](auto& lens) {
        auto offset = max_size - lens.size();
        if(offset > 0)
        {
            std::vector<std::size_t> new_lens(max_size, 1);
            std::copy(lens.begin(), lens.end(), new_lens.begin() + offset);
            lens = new_lens;
        }
    });

    // Compute multidirectional broadcasted shape compatible with all inputs
    std::vector<std::size_t> output_lens(max_size, 1);
    for(std::size_t i = 0; i < max_size; ++i)
    {
        std::for_each(input_lens.begin(), input_lens.end(), [&](const auto& lens) {
            if(lens[i] != 1)
            {
                if(output_lens[i] == 1)
                    output_lens[i] = lens[i];
                else if(output_lens[i] != lens[i])
                    MIGRAPHX_THROW("Parse Mean: Shapes are incompatible for multi-broadcast");
            }
        });
    }

    return output_lens;
}

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

        std::vector<std::vector<std::size_t>> input_lens;
        std::transform(args.begin(),
                       args.end(),
                       std::back_inserter(input_lens),
                       [&](const auto arg) { return arg->get_shape().lens(); });
        auto output_lens = compute_output_lens(input_lens);
        auto output_type = args[0]->get_shape().type();

        auto mean    = args[0];
        auto divisor = info.add_literal(
            migraphx::literal{migraphx::shape{mean->get_shape().type()}, {num_data}});
        divisor = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", output_lens}}), divisor);
        if(not std::equal(mean->get_shape().lens().begin(),
                          mean->get_shape().lens().end(),
                          output_lens.begin(),
                          output_lens.end()))
            mean = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", output_lens}}), mean);

        // Pre-divide each tensor element-wise by n to reduce risk of overflow during summation
        mean = info.add_instruction(migraphx::make_op("div"), mean, divisor);

        std::for_each(args.begin() + 1, args.end(), [&](auto& data_i) {
            if(data_i->get_shape().type() != output_type)
                MIGRAPHX_THROW("Parse Mean: All inputs must have the same data type.");

            auto data_i_lens = data_i->get_shape().lens();
            if(not std::equal(
                   data_i_lens.begin(), data_i_lens.end(), output_lens.begin(), output_lens.end()))
                data_i = info.add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", output_lens}}), data_i);

            // Pre-divide each tensor element-wise by n to reduce risk of overflow during summation
            data_i = info.add_instruction(migraphx::make_op("div"), data_i, divisor);
            mean   = info.add_instruction(migraphx::make_op("add"), mean, data_i);
        });

        return mean;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
