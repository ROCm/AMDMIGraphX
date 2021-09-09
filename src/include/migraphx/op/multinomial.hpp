#ifndef MIGRAPHX_GUARD_OPERATORS_MULTINOMIAL_HPP
#define MIGRAPHX_GUARD_OPERATORS_MULTINOMIAL_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/config.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/tune_axis.hpp>
#include <cmath>
#include <utility>
#include <random>


namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct multinomial
{
    int dtype = 6;
    size_t sample_size = 1;
    float seed = 0.0f; // TODO: auto generate 

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dtype, "dtype"), f(self.sample_size, "sample_size"), f(self.seed, "seed"));
    }

    std::string name() const { return "multinomial"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).only_dims(2);

        if (dtype == 6)
            return {shape::int32_type, {inputs[0].lens()[0], sample_size}};
        if (dtype == 7)
            return {shape::int64_type, {inputs[0].lens()[0], sample_size}};
        else 
            MIGRAPHX_THROW("Invalid output type: " + std::to_string(dtype) + ". Valid types are 6 (INT32) and 7 (INT64).");
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        argument result{output_shape};
        size_t batch_size = output_shape.lens().at(0);
        size_t class_size = args[0].get_shape().lens().at(1);

        args[0].visit([&](auto input) {
            result.visit([&](auto output) {
                par_for(batch_size, [&](auto i) {
                    auto* in_begin = input.data() + (i * class_size);
                    auto* in_end = in_begin + class_size;
                    auto max_iter = std::max_element(in_begin, in_end); // ORT checks for is_finite - needed?
                    
                    std::vector<double> cdf(class_size);
                    std::transform(in_begin, in_end, cdf.begin(), [&](auto logit) { return std::exp(logit - *max_iter); });
                    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin()); // ORT checks for is_finite - needed?

                    auto* out_begin = output.data() + (i * sample_size);
                    auto* out_end = out_begin + sample_size;
                    std::transform(out_begin, out_end, out_begin, [&](auto) {
                        auto idx_iter = std::upper_bound(cdf.begin(), cdf.end(), dis(gen) * cdf.back());
                        return std::distance(cdf.begin(), idx_iter);
                    });
                });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
