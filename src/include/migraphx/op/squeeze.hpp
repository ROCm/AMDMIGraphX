#ifndef MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP

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

struct squeeze
{
    std::vector<int64_t> axes;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::string name() const { return "squeeze"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1).standard();
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();

        // change to support negative axis value
        std::vector<int64_t> tuned_axes(axes.size());
        std::transform(axes.begin(), axes.end(), tuned_axes.begin(), [&](auto i) {
            return i >= 0 ? i : i + old_lens.size();
        });

        if(std::any_of(tuned_axes.begin(), tuned_axes.end(), [&](auto axis) {
               return old_lens[axis] != 1;
           }))
        {
            MIGRAPHX_THROW("squeeze axis dimension should be equal to 1");
        }
        std::vector<std::size_t> new_lens;
        if(tuned_axes.empty())
        {
            std::copy_if(old_lens.begin(),
                         old_lens.end(),
                         std::back_inserter(new_lens),
                         [](auto len) { return len != 1; });
        }
        else
        {
            for(std::size_t i = 0; i < old_lens.size(); i++)
            {
                if(std::find(tuned_axes.begin(), tuned_axes.end(), i) == tuned_axes.end())
                {
                    new_lens.push_back(old_lens[i]);
                }
            }
        }

        if(new_lens.empty())
        {
            return shape{type};
        }
        else
        {
            return shape{type, new_lens};
        }
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
