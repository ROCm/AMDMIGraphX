#ifndef MIGRAPHX_GUARD_OPERATORS_BINARY_HPP
#define MIGRAPHX_GUARD_OPERATORS_BINARY_HPP

#include <migraphx/op/name.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

template <class Derived>
struct binary : op_name<Derived>
{
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs}.has(2).same_type().same_dims();
        const auto& s = inputs.front();
        if(s.scalar() and s.elements() == 1)
            return {s.type()};
        return {s.type(), s.lens()};
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
            if(input1.get_shape().standard() and input2.get_shape().standard())
            {
                std::transform(input1.begin(),
                               input1.end(),
                               input2.begin(),
                               output.begin(),
                               static_cast<const Derived&>(*this).apply());
            }
            else
            {
                shape_for_each(output.get_shape(), [&](const auto& idx) {
                    output(idx.begin(), idx.end()) = static_cast<const Derived&>(*this).apply()(
                        input1(idx.begin(), idx.end()), input2(idx.begin(), idx.end()));
                });
            }
        });
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
