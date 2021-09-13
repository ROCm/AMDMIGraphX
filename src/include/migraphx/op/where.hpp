#ifndef MIGRAPHX_GUARD_OPERATORS_WHERE_HPP
#define MIGRAPHX_GUARD_OPERATORS_WHERE_HPP

#include <array>
#include <migraphx/argument.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/op/normalize_attribute.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct where
{
    std::string name() const { return "where"; }

    value attributes() const { return {{"pointwise", true}, {"point_op", "${0} ? ${1} : ${2}"}}; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).same_dims();
        auto s1 = inputs.at(1);
        auto s2 = inputs.at(2);
        if(s1 == s2 and s1.packed())
        {
            return s1;
        }
        else if(s1.packed() != s2.packed())
        {
            return s1.packed() ? s1 : s2;
        }
        else if(s1.broadcasted() != s2.broadcasted())
        {
            return s1.broadcasted() ? s2.with_lens(s1.lens()) : s1.with_lens(s1.lens());
        }
        else
        {
            return {s1.type(), s1.lens()};
        }
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[1], args[2])([&](auto output, const auto x, const auto y) {
            args[0].visit([&](const auto condition) {
                par_for(output_shape.elements(),
                        [&](auto i) { output[i] = condition[i] ? x[i] : y[i]; });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
