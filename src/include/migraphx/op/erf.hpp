#ifndef MIGRAPHX_GUARD_OPERATORS_ERF_HPP
#define MIGRAPHX_GUARD_OPERATORS_ERF_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct erf : unary<erf>
{
    auto apply() const
    {
        return [](auto x) { return std::erf(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
