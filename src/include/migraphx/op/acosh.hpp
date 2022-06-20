#ifndef MIGRAPHX_GUARD_OPERATORS_ACOSH_HPP
#define MIGRAPHX_GUARD_OPERATORS_ACOSH_HPP

#include <migraphx/op/unary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct acosh : unary<acosh>
{
    auto apply() const
    {
        return [](auto x) { return std::acosh(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
