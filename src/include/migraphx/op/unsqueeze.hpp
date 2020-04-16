#ifndef MIGRAPHX_GUARD_OPERATORS_UNSQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNSQUEEZE_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct unsqueeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "unsqueeze"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard_or_scalar();
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();

        if(input_shape.scalar())
            return shape{type, old_lens};

        std::size_t new_size = old_lens.size() + axes.size();

        // in case of axes to be negative, tune to positive
        std::vector<int64_t> tuned_axes(axes.size());
        std::transform(axes.begin(), axes.end(), tuned_axes.begin(), [new_size](auto i) {
            return i >= 0 ? i : i + new_size;
        });

        std::vector<std::size_t> new_lens(new_size);
        std::size_t p = 0;
        for(std::size_t i = 0; i < new_size; i++)
        {
            if(std::find(tuned_axes.begin(), tuned_axes.end(), i) != tuned_axes.end())
            {
                new_lens[i] = 1;
            }
            else
            {
                new_lens[i] = old_lens[p++];
            }
        }
        return shape{type, new_lens};
    }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.front().data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
