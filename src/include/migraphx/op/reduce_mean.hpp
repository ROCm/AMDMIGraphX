#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_MEAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_MEAN_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_mean : reduce_op<reduce_mean>
{
    reduce_mean() {}
    reduce_mean(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }

    auto output(const shape& s) const
    {
        return [&](auto val) { return val / s.elements(); };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
