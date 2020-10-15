#ifndef MIGRAPHX_GUARD_OPERATORS_LESS_HPP
#define MIGRAPHX_GUARD_OPERATORS_LESS_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct less : binary<less>
{
    auto apply() const
    {
        return [](auto x, auto y) { return x < y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
