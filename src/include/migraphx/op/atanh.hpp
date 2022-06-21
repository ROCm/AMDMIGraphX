#ifndef MIGRAPHX_GUARD_OPERATORS_ATANH_HPP
#define MIGRAPHX_GUARD_OPERATORS_ATANH_HPP

#include <migraphx/op/unary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct atanh : unary<atanh>
{
    auto apply() const
    {
        return [](auto x) { return std::atanh(x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
