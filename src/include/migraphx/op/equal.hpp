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
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims();
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        if(s0 == s1 and s0.packed())
        {
            return {shape::bool_type, s0.lens(), s0.strides()};
        }
        else
        {
            return {shape::bool_type, s0.lens()};
        }
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
