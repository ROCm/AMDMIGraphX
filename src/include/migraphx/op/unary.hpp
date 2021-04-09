#ifndef MIGRAPHX_GUARD_OPERATORS_UNARY_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNARY_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct unary : op_name<Derived>
{
    value base_attributes() const { return {{"pointwise", true}}; }
    value attributes() const { return base_attributes(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(1);
        auto s = inputs.at(0);
        if(s.broadcasted())
        {
            return {s.type(), s.lens()};
        }
        else
        {
            return s.with_lens(s.lens());
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        result.visit([&](auto output) {
            args[0].visit([&](auto input) {
                std::transform(input.begin(),
                               input.end(),
                               output.begin(),
                               static_cast<const Derived&>(*this).apply());

            });
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
