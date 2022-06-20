#ifndef MIGRAPHX_GUARD_OPERATORS_PRELU_HPP
#define MIGRAPHX_GUARD_OPERATORS_PRELU_HPP

#include <migraphx/op/binary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct prelu : binary<prelu>
{
    std::string point_op() const { return "(${0} < 0) ? (${0} * ${1}) : ${0}"; }
    auto apply() const
    {
        return [](auto x, auto slope) { return ((x < 0) ? (x * slope) : x); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
