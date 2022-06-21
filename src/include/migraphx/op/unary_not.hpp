#ifndef MIGRAPHX_GUARD_OPERATORS_UNARY_NOT_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNARY_NOT_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct unary_not : unary<unary_not>
{
    std::string point_function() const { return "!"; }
    auto apply() const
    {
        return [](auto x) { return not x; };
    }

    std::string name() const { return "not"; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
