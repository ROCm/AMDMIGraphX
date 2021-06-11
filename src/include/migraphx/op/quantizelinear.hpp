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
    int axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "quantizelinear"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() == 3)
        {
            return {inputs[2].type(), inputs[0].lens(), inputs[0].strides()};
        }
        return {shape::int8_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto y_zero_point = literal({shape::int8_type, {1}}, {0}).get_argument();
        if(args.size() == 3)
            y_zero_point = args[2];

        auto x       = args[0];
        auto y_scale = args[1];

        auto output_lens = output_shape.lens();
        auto tuned_axis  = tune_axis(output_lens.size(), axis, this->name());
        std::vector<size_t> bcast_strides(output_lens.size(), 0);

        if(y_scale.get_shape().elements() != 1)
            bcast_strides[tuned_axis] = 1;
        migraphx::shape bcast_scales{y_scale.get_shape().type(), output_lens, bcast_strides};
        y_scale                   = y_scale.reshape(bcast_scales);
        bcast_strides[tuned_axis] = 0;

        if(y_zero_point.get_shape().elements() != 1)
            bcast_strides[tuned_axis] = 1;
        migraphx::shape bcast_zeros{y_zero_point.get_shape().type(), output_lens, bcast_strides};
        y_zero_point = y_zero_point.reshape(bcast_zeros);

        argument result{output_shape};
        visit_all(x, y_scale)([&](auto input, auto scales) {
            visit_all(result, y_zero_point)([&](auto output, auto zero_pts) {
                using quant_type  = typename decltype(output)::value_type;
                int64_t min_value = std::numeric_limits<quant_type>::min();
                int64_t max_value = std::numeric_limits<quant_type>::max();
                par_for(output_shape.elements(), [&](auto i) {
                    int64_t quantized = static_cast<int>(std::round(input[i] / scales[i])) +
                                        static_cast<int>(zero_pts[i]);
                    output[i] = std::max(min_value, std::min(max_value, quantized));
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
