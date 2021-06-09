#ifndef MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP
#define MIGRAPHX_GUARD_OPERATORS_DEQUANTIZE_LINEAR_HPP

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
#include <migraphx/par_for.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct dequantizelinear
{
    int axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "dequantizelinear"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        return {shape::float_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto x_zero_point = literal({shape::int8_type, {1}}, {0}).get_argument();
        if (args.size() == 3) 
            x_zero_point = args[2];

        auto x = args[0];
        auto x_scale = args[1];
        argument result{output_shape};
        visit_all(x, x_zero_point)([&](auto input, auto zero_pts) {
            visit_all(result, x_scale)([&](auto output, auto scales) {
                auto num_scales = scales.size();
                auto num_zeros = zero_pts.size();
                par_for(output_shape.elements(), [&](auto i) {
                    auto idx = output_shape.multi(i);
                    auto data = static_cast<int>(input(idx.begin(), idx.end())) - static_cast<int>(zero_pts[idx[axis] % num_zeros]);
                    output(idx.begin(), idx.end()) = static_cast<float>(data) * scales[idx[axis] % num_scales];
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
