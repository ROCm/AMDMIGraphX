#ifndef MIGRAPHX_GUARD_OPERATORS_SIGN_HPP
#define MIGRAPHX_GUARD_OPERATORS_SIGN_HPP

#include <array>
#include <migraphx/op/unary.hpp>
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

struct sign : unary<sign>
{
    auto apply() const
    {
        return [](auto x) { return (x > 0 ? 1 : ((x < 0) ? -1 : 0)); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
