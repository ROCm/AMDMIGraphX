#ifndef MIGRAPHX_GUARD_OPERATORS_UNARY_HPP
#define MIGRAPHX_GUARD_OPERATORS_UNARY_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct unary : op_name<Derived>
{
    std::string point_function() const { return this->name(); }
    std::string point_op() const
    {
        const auto& self = static_cast<const Derived&>(*this);
        auto pf          = self.point_function();
        if(pf.empty())
            return {};
        if(with_char(::ispunct)(pf.front()))
        {
            return pf + "${0}";
        }
        else
        {
            return "${function:" + pf + "}(${0})";
        }
    }
    value base_attributes() const
    {
        const auto& self = static_cast<const Derived&>(*this);
        return {{"pointwise", true}, {"point_op", self.point_op()}};
    }
    value attributes() const { return base_attributes(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(1);
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
                        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
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
