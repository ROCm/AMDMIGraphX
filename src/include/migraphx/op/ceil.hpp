#ifndef MIGRAPHX_GUARD_OPERATORS_CEIL_HPP
#define MIGRAPHX_GUARD_OPERATORS_CEIL_HPP

#include <migraphx/op/unary.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct ceil : unary<ceil>
{
    auto apply() const
    {
        return [](auto x) { return std::ceil(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
