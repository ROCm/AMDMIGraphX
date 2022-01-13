#ifndef MIGRAPHX_GUARD_OPERATORS_CLIP_HPP
#define MIGRAPHX_GUARD_OPERATORS_CLIP_HPP

#include <array>
#include <migraphx/op/unary.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct clip
{
    std::string name() const { return "clip"; }

    value attributes() const
    {
        return {{"pointwise", true},
                {"point_op", "${function:min}(${function:max}(${1}, ${0}), ${2})"}};
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3).same_type().same_dims();
        return inputs.front();
    }

    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        argument result{output_shape};
        visit_all(result, args[0], args[1], args[2])([&](auto output, auto x, auto min, auto max) {
            par_for(output_shape.elements(),
                    [&](auto i) { output[i] = std::min(std::max(min[i], x[i]), max[i]); });
        });

        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
