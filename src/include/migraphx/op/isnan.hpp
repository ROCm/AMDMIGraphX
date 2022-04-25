#ifndef MIGRAPHX_GUARD_OPERATORS_ISNAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ISNAN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct isnan : unary<isnan>
{
    auto apply() const
    {
        return [](auto x) { return std::isnan(x); };
    }

    std::string name() const { return "isnan"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return unary<isnan>::compute_shape(std::move(inputs)).with_type(shape::bool_type);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
