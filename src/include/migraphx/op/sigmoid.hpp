#ifndef MIGRAPHX_GUARD_OPERATORS_SIGMOID_HPP
#define MIGRAPHX_GUARD_OPERATORS_SIGMOID_HPP

#include <array>
#include <migraphx/op/unary.hpp>
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

struct sigmoid : unary<sigmoid>
{
    std::string point_op() const { return "1.f / (1.f + ${function:exp}(-${0}))"; }
    auto apply() const
    {
        return [](auto x) { return 1.f / (1.f + std::exp(-x)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
