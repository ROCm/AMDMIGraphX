#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_multinomial : op_parser<parse_multinomial>
{
    std::vector<op_desc> operators() const { return {{"Multinomial"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int dtype = 6;
        if(contains(info.attributes, "dtype"))
            dtype = info.attributes.at("dtype").i();
        shape::type_t output_type = get_type(dtype);

        size_t sample_size = 1;
        if(contains(info.attributes, "sample_size"))
            sample_size = info.attributes.at("sample_size").i();

        // Subtract the per-batch maximum log-probability, making the per-batch max 0
        auto maxes =
            info.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), args[0]);
        auto mb_maxes = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", args[0]->get_shape().lens()}}),
            maxes);
        auto cdf = info.add_instruction(migraphx::make_op("sub"), args[0], mb_maxes);
        // Take the element-wise exponent to get probabilities in the range (0, 1]
        cdf = info.add_instruction(migraphx::make_op("exp"), cdf);
        // Compute the cumulative density function
        cdf = info.add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

        // Pre-compute random distribution
        std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if(contains(info.attributes, "seed"))
            gen.seed(info.attributes.at("seed").f());

        std::uniform_real_distribution<> dis(0.0, 1.0);
        size_t batch_size = args[0]->get_shape().lens().front();
        migraphx::shape dist_shape{migraphx::shape::float_type, {batch_size, sample_size}};

        std::vector<float> random_dist(batch_size * sample_size);
        std::generate(random_dist.begin(), random_dist.end(), [&]() { return dis(gen); });
        auto dist_lit = info.add_literal(migraphx::literal{dist_shape, random_dist});

        return info.add_instruction(
            migraphx::make_op("multinomial", {{"dtype", output_type}}), cdf, dist_lit);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
