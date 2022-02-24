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
        check_shapes{inputs, *this}.has(1);
        auto i = inputs.at(0);
        auto s = shape{shape::bool_type, i.lens(), i.strides()};
        if(s.scalar())
        {
            return s;
        }
        else if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
