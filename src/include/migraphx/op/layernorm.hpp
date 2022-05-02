#ifndef MIGRAPHX_GUARD_OPERATORS_LAYERNORMALIZATION_HPP
#define MIGRAPHX_GUARD_OPERATORS_LAYERNORMALIZATION_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/par_for.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct layernorm
{
    float epsilon = 1e-3;
    int64_t axis  = -1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.epsilon, "epsilon"), f(self.axis, "axis"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "layernorm"; }

    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() == 2)
        {
            if(inputs.at(1).lens().front() != inputs.front().lens().at(axis))
                MIGRAPHX_THROW("LAYERNORM: weights have wrong shape");
        }
        if(inputs.size() == 3)
        {
            if(inputs.at(2).lens().front() != inputs.front().lens().at(axis))
                MIGRAPHX_THROW("LAYERNORM: bias has wrong shape");
        }

        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto x_lens     = args.front().get_shape().lens();
        auto norm_count = std::accumulate(
            x_lens.begin(), x_lens.begin() + axis, std::size_t{1}, std::multiplies<std::size_t>());
        auto norm_size = std::accumulate(
            x_lens.begin() + axis, x_lens.end(), std::size_t{1}, std::multiplies<std::size_t>());

        /* std::vector<std::size_t> mean_inv_std_dev_dim(x_lens.size());
        for (std::size_t i = 0; i < x_lens.size(); ++i)
        {
            if (i < axis)
                mean_inv_std_dev_dim.at(i) = x_lens[i];
            else
                mean_inv_std_dev_dim.at(i) = 1;

        } */
        if(args.size() == 3)
        {
            visit_all(result, args[0], args[1], args[2])(
                [&](auto output, auto data, auto weights, auto bias) {
                    par_for(norm_count, [&](auto idx) {
                        auto offset        = idx * norm_size;
                        double mean        = 0;
                        double mean_square = 0;
                        for(std::size_t i = 0; i < norm_size; ++i)
                        {
                            mean += data[offset + i];
                            mean_square += data[offset + i] * data[offset + i];
                        }
                        mean /= norm_size;
                        mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
                        for(std::size_t i = 0; i < norm_size; ++i)
                        {
                            if(args.size() == 3)
                                output[offset + i] =
                                    (data[offset + i] - mean) / mean_square * weights[i] + bias[i];
                            else
                                output[offset + i] =
                                    (data[offset + i] - mean) / mean_square * weights[i];
                        }
                    });
                });
        }
        else
        {
            visit_all(result, args[0])([&](auto output, auto data) {
                par_for(norm_count, [&](auto idx) {
                    auto offset        = idx * norm_size;
                    double mean        = 0;
                    double mean_square = 0;
                    for(std::size_t i = 0; i < norm_size; ++i)
                    {
                        mean += data[offset + i];
                        mean_square += data[offset + i] * data[offset + i];
                    }
                    mean /= norm_size;
                    mean_square = sqrt(mean_square / norm_size - mean * mean + epsilon);
                    for(std::size_t i = 0; i < norm_size; ++i)
                    {
                        output[offset + i] = (data[offset + i] - mean) / mean_square;
                        // scale and bias handled by onnx parser
                    }
                });
            });
        }

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
