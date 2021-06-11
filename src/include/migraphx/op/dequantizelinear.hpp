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
    int axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "dequantizelinear"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return {shape::float_type, inputs[0].lens(), inputs[0].strides()};
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        auto x_zero_point = literal({shape::int8_type, {1}}, {0}).get_argument();
        if(args.size() == 3)
            x_zero_point = args[2];

        auto x       = args[0];
        auto x_scale = args[1];

        auto output_lens = output_shape.lens();
        auto tuned_axis  = tune_axis(output_lens.size(), axis, this->name());
        std::vector<size_t> bcast_strides(output_lens.size(), 0);

        if(x_scale.get_shape().elements() != 1)
            bcast_strides[tuned_axis] = 1;
        migraphx::shape bcast_scales{x_scale.get_shape().type(), output_lens, bcast_strides};
        x_scale                   = x_scale.reshape(bcast_scales);
        bcast_strides[tuned_axis] = 0;

        if(x_zero_point.get_shape().elements() != 1)
            bcast_strides[tuned_axis] = 1;
        migraphx::shape bcast_zeros{x_zero_point.get_shape().type(), output_lens, bcast_strides};
        x_zero_point = x_zero_point.reshape(bcast_zeros);

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
