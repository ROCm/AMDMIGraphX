#ifndef MIGRAPHX_GUARD_OPERATORS_SCATTERND_NONE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCATTERND_NONE_HPP

#include <migraphx/op/scatternd_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scatternd_none : scatternd_op<scatternd_none>
{
    scatternd_none() {}

    auto reduction() const
    {
        return [](auto& x, const auto& y) { x = y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
