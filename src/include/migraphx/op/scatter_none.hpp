#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_NONE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_NONE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <migraphx/op/scatter.hpp>
#include <cmath>
#include <utility>

// Scatter op. with "none" as the reduction function (just copies the value).  This is identical to the
// previously existing Scatter op.
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatter_none : scatter<scatter_none>
{
    // reduction (pointwise operation) is called by the parent struct's compute() method.
    // It works much like a virtual function overload.
    // For the scatter operators, there are three different reduction functions.
    auto reduction() const
    {
        return [](auto& x, const auto& y) { x = y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
