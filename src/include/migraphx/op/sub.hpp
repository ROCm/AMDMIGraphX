#ifndef MIGRAPHX_GUARD_OPERATORS_SUB_HPP
#define MIGRAPHX_GUARD_OPERATORS_SUB_HPP

#include <array>
#include <migraphx/op/binary.hpp>
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

struct sub : binary<sub>
{
    auto apply() const
    {
        return [](auto x, auto y) { return x - y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
