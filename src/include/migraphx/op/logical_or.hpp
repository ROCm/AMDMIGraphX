#ifndef MIGRAPHX_GUARD_OPERATORS_LOGICAL_OR_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOGICAL_OR_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct logical_or : binary<logical_or>
{
    std::string point_function() const { return "||"; }
    auto apply() const
    {
        return [](auto x, auto y) { return static_cast<bool>(x) or static_cast<bool>(y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
