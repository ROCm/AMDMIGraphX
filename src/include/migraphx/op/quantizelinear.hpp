#ifndef MIGRAPHX_GUARD_OPERATORS_QUANTIZE_LINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANTIZE_LINEAR_HPP

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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct quantizelinear
{
    std::string name() const { return "quantizelinear"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() == 3)
        {
            return {inputs[2].type(), inputs[0].lens(), inputs[0].strides()};
        }
        return {shape::uint8_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto x       = args.at(0);
        auto y_scale = args.at(1);
        std::vector<int8_t> zeros(output_shape.elements(), 0);
        argument y_zero_point{output_shape, zeros.data()};
        if(args.size() == 3)
        {
            y_zero_point = args.at(2);
        }

        argument result{output_shape};
        visit_all(result, y_zero_point)([&](auto output, auto zero_pts) {
            x.visit([&](auto input) {
                y_scale.visit([&](auto scales) {
                    using quant_type = typename decltype(output)::value_type;
                    auto min_value   = std::numeric_limits<quant_type>::min();
                    auto max_value   = std::numeric_limits<quant_type>::max();
                    par_for(output_shape.elements(), [&](auto i) {
                        int64_t quantized = static_cast<int64_t>(std::round(input[i] / scales[i])) +
                                            static_cast<int64_t>(zero_pts[i]);
                        output[i] = std::max(static_cast<int64_t>(min_value),
                                             std::min(static_cast<int64_t>(max_value), quantized));
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
