#ifndef MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP
#define MIGRAPHX_GUARD_OPERATORS_ALLOCATE_HPP

#include <array>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct allocate
{
    std::string name() const { return "allocate"; }
    shape compute_shape(std::vector<shape> inputs) const
    {
        return inputs.front();
    }
    argument compute(const shape& output_shape, std::vector<argument> args) const
    {
        return {output_shape};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
