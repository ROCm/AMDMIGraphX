#ifndef MIGRAPHX_GUARD_OPERATORS_RSQRT_HPP
#define MIGRAPHX_GUARD_OPERATORS_RSQRT_HPP

#include <migraphx/op/unary.hpp>
#include <cmath>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct rsqrt : unary<rsqrt>
{
    auto apply() const
    {
        return [](auto x) { return 1 / std::sqrt(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
