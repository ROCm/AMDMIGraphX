#ifndef MIGRAPHX_GUARD_OPERATORS_SQRT_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQRT_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sqrt : unary<sqrt>
{
    auto apply() const
    {
        return [](auto x) { return std::sqrt(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
