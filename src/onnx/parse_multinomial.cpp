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

        float seed = 0.0; // generate randomly if not specified
        if(contains(info.attributes, "seed"))
            seed = parser.parse_value(info.attributes.at("seed")).at<float>();

        auto input_lens = args[0]->get_shape().lens();
        if(input_lens.size() != 2)
            MIGRAPHX_THROW("Incorrect shape for input tensor. Expected {batch_size, class_size}");

        std::vector<double> logits;
        std::vector<std::vector<int32_t>> output;
        auto args0 = args[0]->eval();
        args0.visit([&](auto input) { logits.assign(input.begin(), input.end()); });

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for(size_t i = 0; i < input_lens[0]; ++i)
        {
            std::vector<double> cdf(input_lens[1], 0);
            double maxx = std::numeric_limits<float>::lowest();
            for(size_t j = 0; j < input_lens[1]; ++j)
            {
                auto idx = (i * input_lens[0]) + input_lens[1];
                maxx     = std::max(maxx, logits[idx]);
            }

            for(size_t j = 0; j < input_lens[1]; ++j)
            {
                auto idx = (i * input_lens[0]) + input_lens[1];
                cdf[j]   = std::exp(logits[idx] - maxx);
            }

            double running_total = 0;
            for(size_t j = 0; j < input_lens[1]; ++j)
            {
                running_total += cdf[j];
                cdf[j] = running_total;
            }

            for(size_t j = 0; j < sample_size; ++j)
            {
                double to_find = dist(gen) * running_total;
                auto iter      = std::upper_bound(cdf.begin(), cdf.end(), to_find);
                output[i][j]   = static_cast<int64_t>(std::distance(cdf.begin(), iter));
                std::cout << output[i][j] << ", ";
            }
            std::cout << std::endl;
        }

        std::vector<int32_t> output_1d;
        for(auto& row : output)
            for(auto& it : row)
                output_1d.push_back(it);
        shape output_shape{shape::int32_type, {input_lens[0], sample_size}};
        return info.add_literal(migraphx::literal(output_shape, output_1d));
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
