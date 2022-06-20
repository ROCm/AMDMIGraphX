#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTER_ADD_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTER_ADD_HPP

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

// Scatter op. with "add" function as reduction.
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatter_add : scatter<scatter_add>
{
    // reduction (pointwise operation) is called by the parent struct's compute() method.
    // It works much like a virtual function overload.
    // For the scatter methods, there are three different reduction functions.
    auto reduction() const
    {
        return [](auto& x, const auto& y) { x += y; };
    }

    // name of this struct is automatically assigned by the op_name<>
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
