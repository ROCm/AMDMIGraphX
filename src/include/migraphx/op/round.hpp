#ifndef MIGRAPHX_GUARD_OPERATORS_ROUND_HPP
#define MIGRAPHX_GUARD_OPERATORS_ROUND_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct round : unary<round>
{
    auto apply() const
    {
        return [](auto x) { return std::round(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
