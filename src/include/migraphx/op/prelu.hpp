#ifndef MIGRAPHX_GUARD_OPERATORS_PRELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_PRELU_HPP

#include <migraphx/op/binary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct prelu : binary<prelu>
{
    auto apply() const
    {
        return [](auto x, auto slope) { return ((x < 0) ? (x * slope) : x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
