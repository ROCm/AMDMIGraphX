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
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int dtype = 6; // 6=in32, 7=int64
        if(contains(info.attributes, "dtype"))
            dtype = parser.parse_value(info.attributes.at("dtype")).at<int>();

        size_t sample_size = 1;
        if(contains(info.attributes, "sample_size"))
            sample_size = parser.parse_value(info.attributes.at("sample_size")).at<int>();

        float seed = static_cast<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if(contains(info.attributes, "seed"))
            seed = parser.parse_value(info.attributes.at("seed")).at<float>();

        // Compute cumulative density function
        auto maxes =
            info.add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), args[0]);
        auto mb_maxes = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", args[0]->get_shape().lens()}}),
            maxes);
        auto cdf = info.add_instruction(migraphx::make_op("sub"), args[0], mb_maxes);
        cdf      = info.add_instruction(migraphx::make_op("exp"), cdf);
        cdf      = info.add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

        // Pre-compute random distribution
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        size_t batch_size = args[0]->get_shape().lens().front();
        migraphx::shape dist_shape{migraphx::shape::float_type, {batch_size, sample_size}};

        std::vector<float> random_dist(batch_size * sample_size);
        std::transform(random_dist.begin(), random_dist.end(), random_dist.begin(), [&](auto) {
            return dis(gen);
        });
        auto dist_lit = info.add_literal(migraphx::literal{dist_shape, random_dist});

        return info.add_instruction(
            migraphx::make_op("multinomial", {{"dtype", dtype}}), cdf, dist_lit);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
