#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTERND_MUL_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTERND_MUL_HPP

#include <migraphx/op/scatternd_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatternd_mul : scatternd_op<scatternd_mul>
{
    scatternd_mul() {}

    auto reduction() const
    {
        return [](auto& x, const auto& y) { x *= y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
