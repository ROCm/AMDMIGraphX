#ifndef MIGRAPHX_GUARD_OPERATORS_UNSQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNSQUEEZE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/lifetime.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct unsqueeze
{
    std::vector<int64_t> axes;
    std::vector<int64_t> steps;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"), f(self.steps, "steps"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axes"] =
            value::array{normalize_attribute::include_min, normalize_attribute::use_output};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "unsqueeze"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();
        auto old_strides = input_shape.strides();
        if(input_shape.scalar())
        {
            if(old_lens.size() == 1 and old_lens.front() == 1)
                return shape{type, old_lens};
            else
                MIGRAPHX_THROW("UNSQUEEZE: Input must be a scalar");
        }

        std::size_t new_size = old_lens.size() + axes.size();

        std::vector<std::size_t> new_lens(new_size);
        std::vector<std::size_t> new_strides(new_size);
        std::size_t p = 0;
        for(auto i : range(new_size))
        {
            auto axis_idx = std::find(axes.begin(), axes.end(), i) - axes.begin();
            if(axis_idx < axes.size())
            {
                std::int64_t step = 1;
                if(axis_idx < steps.size())
                    step = steps[axis_idx];
                if(step == 0)
                    MIGRAPHX_THROW("UNSQUEEZE: step must be non-zero");
                new_lens[i] = step;
                if(p < old_strides.size())
                {
                    if((old_lens[p] % step) != 0)
                        MIGRAPHX_THROW("UNSQUEEZE: Axis dimenstion is not divisible by step");
                    old_lens[p] /= step;
                    new_strides[i] = old_strides[p] * old_lens[p];
                }
                else
                {
                    if(step != 1)
                        MIGRAPHX_THROW("UNSQUEEZE: Step must be 1 for extra axes");
                    new_strides[i] = 1;
                }
            }
            else
            {
                new_lens[i]    = old_lens[p];
                new_strides[i] = old_strides[p++];
            }
        }
        return shape{type, new_lens, new_strides};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return args[0].reshape(output_shape);
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
