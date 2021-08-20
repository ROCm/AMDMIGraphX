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

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.same_dims();
        return inputs[1];
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[1], args[2])([&](auto output, const auto x, const auto y) {
            args[0].visit([&](const auto cond) {
                par_for(output_shape.elements(),
                        [&](auto i) { output[i] = cond[i] ? x[i] : y[i]; });
            });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
