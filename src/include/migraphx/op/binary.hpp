#ifndef MIGRAPHX_GUARD_OPERATORS_BINARY_HPP
#define MIGRAPHX_GUARD_OPERATORS_BINARY_HPP

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
struct binary : op_name<Derived>
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
            return "${0} " + pf + " ${1}";
        }
        else
        {
            return "${function:" + pf + "}(${0}, ${1})";
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
        check_shapes{inputs, static_cast<const Derived&>(*this)}.has(2).same_type().same_dims();
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        if(s0 == s1 and s0.packed())
        {
            return s0;
        }
        else if(s0.packed() != s1.packed())
        {
            return s0.packed() ? s0 : s1;
        }
        else if(s0.broadcasted() != s1.broadcasted())
        {
            return s0.broadcasted() ? s1.with_lens(s0.lens()) : s0.with_lens(s0.lens());
        }
        else
        {
            return {s0.type(), s0.lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
            std::transform(input1.begin(),
                           input1.end(),
                           input2.begin(),
                           output.begin(),
                           static_cast<const Derived&>(*this).apply());
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
