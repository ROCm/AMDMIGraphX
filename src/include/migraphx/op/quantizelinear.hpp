#ifndef MIGRAPHX_GUARD_OPERATORS_QUANTIZE_LINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_QUANTIZE_LINEAR_HPP

#include <array>
#include <migraphx/op/common.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/op/tanh.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <migraphx/par_for.hpp>

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
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        if(inputs.size() == 3)
        {
            return {inputs[2].type(), inputs[0].lens(), inputs[0].strides()};
        }
        return {shape::int8_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto quant_type = shape::int8_type;
        auto y_zero_point = literal({quant_type, {1}}, {0}).get_argument();
        if (args.size() == 3) 
        {
            quant_type = args[2].get_shape().type();
            y_zero_point = args[2];
        }

        int max_quant = 255;
        int min_quant = 0;
        if (quant_type == shape::int8_type)
        {
            max_quant = 127;
            min_quant = -128;
        }

        auto x = args[0];
        auto y_scale = args[1];
        argument result{output_shape};
        visit_all(x, y_scale)([&](auto input, auto scales) {
            visit_all(result, y_zero_point)([&](auto output, auto zero_pts) {
                auto num_scales = scales.size();
                auto num_zeros = zero_pts.size();
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = output_shape.multi(i);
                    float data = std::round(input(idx.begin(), idx.end()) / scales[idx[axis] % num_scales]);
                    auto int32_data = static_cast<int>(data) + static_cast<int>(zero_pts[idx[axis] % num_zeros]);
                    int32_data = std::max(min_quant, std::min(max_quant, int32_data));
                    output(idx.begin(), idx.end()) = int32_data;
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
