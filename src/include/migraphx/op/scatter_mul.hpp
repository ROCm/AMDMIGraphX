#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_MUL_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_MUL_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>
#include <migraphx/op/scatter.hpp>

// Scatter op. with "multiply" as the reduction function.
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatter_mul : scatter<scatter_mul>
{
    // reduction (pointwise operation) is called by the parent struct's compute() method.
    // It works much like a virtual function overload.
    // For the scatter operators, there are three different reduction functions.
    auto reduction() const
    {
        return [](auto& x, const auto& y) { x *= y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
