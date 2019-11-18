#ifndef MIGRAPHX_GUARD_OPERATORS_FLOOR_HPP
#define MIGRAPHX_GUARD_OPERATORS_FLOOR_HPP

#include <migraphx/op/unary.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct floor : unary<floor>
{
    auto apply() const
    {
        return [](auto x) { return std::floor(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
