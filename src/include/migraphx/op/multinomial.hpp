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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.dtype, "dtype"));
    }

    std::string name() const { return "multinomial"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(2).only_dims(2);
        size_t sample_size = inputs.back().lens().back();

        if(dtype == 6)
            return {shape::int32_type, {inputs[0].lens()[0], sample_size}};
        if(dtype == 7)
            return {shape::int64_type, {inputs[0].lens()[0], sample_size}};
        else
            MIGRAPHX_THROW("Invalid output type: " + std::to_string(dtype) +
                           ". Valid types are 6 (INT32) and 7 (INT64).");
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        size_t batch_size  = output_shape.lens().front();
        size_t class_size  = args[0].get_shape().lens().back();
        size_t sample_size = output_shape.lens().back();

        visit_all(args[0], args[1])([&](auto cdf, auto dist) {
            result.visit([&](auto output) {
                par_for(batch_size * sample_size, [&](auto i) {
                    auto idx        = args[1].get_shape().multi(i);
                    auto* cdf_begin = cdf.data() + (idx[0] * class_size);
                    auto* cdf_end   = cdf_begin + class_size;
                    auto* sample_iter =
                        std::upper_bound(cdf_begin, cdf_end, dist[i] * *(cdf_end - 1));
                    output[i] = std::distance(cdf_begin, sample_iter);
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
