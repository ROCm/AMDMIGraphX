#ifndef MIGRAPHX_GUARD_OPERATORS_SQDIFF_HPP
#define MIGRAPHX_GUARD_OPERATORS_SQDIFF_HPP

#include <migraphx/op/binary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sqdiff : binary<sqdiff>
{
    auto apply() const
    {
        return [](auto x, auto y) { return (x - y) * (x - y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
