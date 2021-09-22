#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <random>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_randomnormal_ops : op_parser<parse_randomnormal_ops>
{
    const std::set<shape::type_t> valid_types = {
        shape::float_type, shape::half_type, shape::double_type};

    std::vector<op_desc> operators() const { return {{"RandomNormal"}, {"RandomNormalLike"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        int dtype      = 1;
        bool use_dtype = false;
        if(contains(info.attributes, "dtype"))
        {
            dtype     = parser.parse_value(info.attributes.at("dtype")).at<int>();
            use_dtype = true;
        }
        shape::type_t out_type = get_type(dtype);
        if(valid_types.find(out_type) == valid_types.end())
            MIGRAPHX_THROW(opd.op_name + ": invalid output type: " + std::to_string(dtype) +
                           ". Valid types are 1 (float), 10 (half), and 11 (double).");

        float mean = 0.0;
        if(contains(info.attributes, "mean"))
            mean = parser.parse_value(info.attributes.at("mean")).at<float>();

        float scale = 1.0;
        if(contains(info.attributes, "scale"))
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();

        float seed = static_cast<float>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if(contains(info.attributes, "seed"))
            seed = parser.parse_value(info.attributes.at("seed")).at<float>();

        shape out_shape;
        if(contains(info.attributes, "shape"))
        {
            std::vector<int> out_lens;
            literal ls = parser.parse_value(info.attributes.at("shape"));
            ls.visit([&](auto s) { out_lens.assign(s.begin(), s.end()); });
            out_shape = shape{out_type, out_lens};
        }
        else if(args.size() == 1)
        {
            if(valid_types.find(args[0]->get_shape().type()) == valid_types.end())
                use_dtype = true;
            out_shape =
                use_dtype ? shape{out_type, args[0]->get_shape().lens()} : args[0]->get_shape();
        }
        else
        {
            MIGRAPHX_THROW(opd.op_name +
                           ": cannot deduce shape without shape attribute or argument.");
        }

        std::mt19937 gen{seed};
        std::normal_distribution<> d{mean, scale};
        std::vector<double> rand_vals(out_shape.elements());
        std::transform(
            rand_vals.begin(), rand_vals.end(), rand_vals.begin(), [&](auto) { return d(gen); });

        return info.add_literal(literal{out_shape, rand_vals});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
