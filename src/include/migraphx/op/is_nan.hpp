#ifndef MIGRAPHX_GUARD_OPERATORS_ISNAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_ISNAN_HPP

#include <migraphx/op/unary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct is_nan : unary<is_nan>
{
    std::string point_function() const { return "is_nan"; }
    auto apply() const
    {
        return [](auto x) { return std::isnan(x); };
    }

    std::string name() const { return "is_nan"; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
