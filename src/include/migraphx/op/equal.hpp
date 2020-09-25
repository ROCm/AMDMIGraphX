#ifndef MIGRAPHX_GUARD_OPERATORS_EQUAL_HPP
#define MIGRAPHX_GUARD_OPERATORS_EQUAL_HPP

#include <migraphx/op/binary.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct equal : binary<equal>
{
    value attributes() const
    {
        auto a           = base_attributes();
        a["commutative"] = true;
        return a;
    }
    auto apply() const
    {
        return [](auto x, auto y) { return float_equal(x, y); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
