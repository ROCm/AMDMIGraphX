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
    uint64_t axis = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axis, "axis"));
    }

    std::string name() const { return "flatten"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        auto&& lens = inputs.front().lens();

        if(axis > lens.size())
        {
            MIGRAPHX_THROW("axis for flatten must be less than tensor rank");
        }
        auto x =
            std::accumulate(lens.begin(), lens.begin() + axis, std::size_t{1}, std::multiplies<>{});
        auto y =
            std::accumulate(lens.begin() + axis, lens.end(), std::size_t{1}, std::multiplies<>{});
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
