#ifndef MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQUEEZE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/lifetime.hpp>
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

    value attributes() const
    {
        value normalize;
        normalize["axes"] = value::array{normalize_attribute::include_min};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "squeeze"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();
        auto old_strides = input_shape.strides();
        if(std::any_of(axes.begin(), axes.end(), [&](auto axis) { return old_lens[axis] != 1; }))
        {
            MIGRAPHX_THROW("squeeze axis dimension should be equal to 1");
        }
        std::vector<std::size_t> new_lens;
        std::vector<std::size_t> new_strides;
        if(axes.empty())
        {
            for(auto i : range(old_lens.size()))
            {
                if(old_lens[i] != 1)
                {
                    new_lens.push_back(old_lens[i]);
                    new_strides.push_back(old_strides[i]);
                }
            }
        }
        else
        {
            for(auto i : range(old_lens.size()))
            {
                if(std::find(axes.begin(), axes.end(), i) == axes.end())
                {
                    new_lens.push_back(old_lens[i]);
                    new_strides.push_back(old_strides[i]);
                }
            }
        }
        if(new_lens.empty())
        {
            return shape{type};
        }
        else
        {
            return shape{type, new_lens, new_strides};
        }
    }

    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return args[0].reshape(output_shape);
    }
    lifetime get_lifetime() const { return lifetime::borrow; }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
