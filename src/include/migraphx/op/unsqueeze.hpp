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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
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
        check_shapes{inputs, *this}.has(1).standard_or_scalar();
        auto input_shape = inputs[0];
        auto type        = input_shape.type();
        auto old_lens    = input_shape.lens();

        if(input_shape.scalar())
        {
            if(old_lens.size() == 1 and old_lens.front() == 1)
                return shape{type, old_lens};
            else
                MIGRAPHX_THROW("UNSQUEEZE: Input must be a scalar");
        }

        std::size_t new_size = old_lens.size() + axes.size();

        std::vector<std::size_t> new_lens(new_size);
        std::size_t p = 0;
        for(std::size_t i = 0; i < new_size; i++)
        {
            if(std::find(axes.begin(), axes.end(), i) != axes.end())
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
        return args[0].reshape(output_shape);
    }
    lifetime get_lifetime() const { return lifetime::borrow; }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
