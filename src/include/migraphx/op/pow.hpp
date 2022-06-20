#ifndef MIGRAPHX_GUARD_OPERATORS_POW_HPP
#define MIGRAPHX_GUARD_OPERATORS_POW_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct pow : binary<pow>
{
    auto apply() const
    {
        return [](auto x, auto y) { return std::pow(x, y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
