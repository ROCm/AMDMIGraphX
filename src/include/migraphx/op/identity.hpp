#ifndef MIGRAPHX_GUARD_OPERATORS_IDENTITY_HPP
#define MIGRAPHX_GUARD_OPERATORS_IDENTITY_HPP

#include <array>
#include <migraphx/operation.hpp>
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

struct identity
{
    std::string name() const { return "identity"; }
    shape compute_shape(std::vector<shape> inputs) const { return inputs.at(0); }
    argument compute(shape output_shape, std::vector<argument> args) const
    {
        return {std::move(output_shape), std::move(args.at(0).data)};
    }
    std::ptrdiff_t output_alias(const std::vector<shape>&) const { return 0; }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
