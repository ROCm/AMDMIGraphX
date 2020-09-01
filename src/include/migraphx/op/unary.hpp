#ifndef MIGRAPHX_GUARD_OPERATORS_UNARY_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNARY_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct unary : op_name<Derived>
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(1);
        auto s = inputs.at(0);
        if(s.packed())
        {
            return s;
        }
        else
        {
            return {s.type(), s.lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto in_shape = args[0].get_shape();
        if(in_shape.packed())
        {
            shape std_in_shape{in_shape.type(), in_shape.lens()};
            shape std_out_shape{output_shape.type(), output_shape.lens()};
            argument arg_in{std_in_shape, args[0].data()};
            argument arg_out{std_out_shape, result.data()};
            arg_out.visit([&](auto output) {
                arg_in.visit([&](auto input) {
                    std::transform(input.begin(),
                                   input.end(),
                                   output.begin(),
                                   static_cast<const Derived&>(*this).apply());

                });
            });
        }
        else
        {
            result.visit([&](auto output) {
                args[0].visit([&](auto input) {
                    shape_for_each(output.get_shape(), [&](const auto& idx) {
                        output(idx.begin(), idx.end()) = static_cast<const Derived&>(*this).apply()(
                            input(idx.begin(), idx.end()));
                    });
                });
            });
        }

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
