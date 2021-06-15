#ifndef MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP

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

struct dequantizelinear
{
    std::string name() const { return "dequantizelinear"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return {shape::float_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto x       = args.at(0);
        auto x_scale = args.at(1);
        std::vector<int8_t> zeros(output_shape.elements(), 0);
        argument x_zero_point{{x.get_shape().type(), output_shape.lens()}, zeros.data()};
        if(args.size() == 3)
        {
            x_zero_point = args.at(2);
        }

        argument result{output_shape};
        visit_all(x, x_zero_point)([&](auto input, auto zero_pts) {
            visit_all(result, x_scale)([&](auto output, auto scales) {
                par_for(output_shape.elements(), [&](auto i) {
                    output[i] = static_cast<double>(static_cast<int64_t>(input[i]) -
                                                    static_cast<int64_t>(zero_pts[i])) *
                                scales[i];
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
