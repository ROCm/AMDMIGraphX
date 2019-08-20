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
        auto s0 = inputs.at(0);
        auto s1 = inputs.at(1);
        if(s0 == s1 and s0.packed())
        {
            return s0;
        }
        else
        {
            return {s0.type(), s0.lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        auto s1 = args[0].get_shape();
        auto s2 = args[1].get_shape();
        if(s1 == s2 and s1.packed())
        {
            shape std_shape{s1.type(), s1.lens()};
            argument std_result{std_shape, result.data()};
            argument std_arg0{std_shape, args[0].data()};
            argument std_arg1{std_shape, args[1].data()};
            visit_all(std_result, std_arg0, std_arg1)([&](auto output, auto input1, auto input2) {
                std::transform(input1.begin(),
                               input1.end(),
                               input2.begin(),
                               output.begin(),
                               static_cast<const Derived&>(*this).apply());
            });
        }
        else
        {
            visit_all(result, args[0], args[1])([&](auto output, auto input1, auto input2) {
                shape_for_each(output.get_shape(), [&](const auto& idx) {
                    output(idx.begin(), idx.end()) = static_cast<const Derived&>(*this).apply()(
                        input1(idx.begin(), idx.end()), input2(idx.begin(), idx.end()));
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
