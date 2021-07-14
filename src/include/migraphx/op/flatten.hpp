#ifndef MIGRAPHX_GUARD_OPERATORS_FLATTEN_HPP
#define MIGRAPHX_GUARD_OPERATORS_FLATTEN_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
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

struct flatten
{
    int64_t axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    value attributes() const
    {
        value normalize;
        normalize["axis"] =
            value::array{normalize_attribute::include_min, normalize_attribute::include_max};
        return {{"normalize_axes", normalize}};
    }

    std::string name() const { return "flatten"; }
    shape normalize_compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(1);
        auto&& lens = inputs.front().lens();
        auto x =
            std::accumulate(lens.begin(), lens.begin() + axis, std::size_t{1}, std::multiplies<>{});
        auto y =
            std::accumulate(lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<>{});
        return {inputs.at(0).type(), {x, y}};
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
