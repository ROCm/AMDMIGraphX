#ifndef MIGRAPHX_GUARD_OPERATORS_FLATTEN_HPP
#define MIGRAPHX_GUARD_OPERATORS_FLATTEN_HPP

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

struct flatten
{
    int64_t axis = 1;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "flatten"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        auto&& lens   = inputs.front().lens();
        int64_t n_dim = static_cast<int64_t>(lens.size());
        if(axis > n_dim or axis < -n_dim)
        {
            MIGRAPHX_THROW("FLATTEN: axis for flatten is out of range");
        }

        auto tuned_axis = (axis < 0) ? axis + n_dim : axis;

        auto x = std::accumulate(
            lens.begin(), lens.begin() + tuned_axis, std::size_t{1}, std::multiplies<>{});
        auto y = std::accumulate(
            lens.begin() + tuned_axis, lens.end(), std::size_t{1}, std::multiplies<>{});
        return {inputs.at(0).type(), {x, y}};
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
