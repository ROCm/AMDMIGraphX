#ifndef MIGRAPHX_GUARD_OPERATORS_STEP_HPP
#define MIGRAPHX_GUARD_OPERATORS_STEP_HPP

#include "migraphx/stringutils.hpp"
#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/lifetime.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct step
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
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "step"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input   = inputs.at(0);
        auto in_lens = input.lens();
        auto t       = input.type();

        if(axes.size() != steps.size())
        {
            MIGRAPHX_THROW("STEP: attribute axes {" + to_string_range(axes) +
                           "} has different dimensions from step {" + to_string_range(steps) +
                           "}.");
        }

        if(std::any_of(axes.begin(), axes.end(), [&](auto axis) { return axis >= in_lens.size(); }))
        {
            MIGRAPHX_THROW("STEP: axis value is out of range!");
        }

        auto lens    = in_lens;
        auto strides = input.strides();
        for(auto i : range(axes.size()))
        {
            auto axis  = axes[i];
            auto step  = steps[i];
            lens[axis] = (in_lens[axis] + step - 1) / step;
            strides[axis] *= step;
        }

        return {t, lens, strides};
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
