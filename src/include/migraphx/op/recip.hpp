#ifndef MIGRAPHX_GUARD_OPERATORS_RECIP_HPP
#define MIGRAPHX_GUARD_OPERATORS_RECIP_HPP

#include <migraphx/op/unary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct recip : unary<recip>
{
    auto apply() const
    {
        return [](auto x) { return 1 / x; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
