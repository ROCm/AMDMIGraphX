#ifndef MIGRAPHX_GUARD_OPERATORS_MEAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_MEAN_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct mean_op
{
    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }

    auto init() const { return zero(); }

    auto input() const
    {
        return [&](auto val) { return val; };
    }

    auto output(const shape& s) const
    {
        return [&](auto val) { return val / s.elements(); };
    }
};

struct reduce_mean : reduce_op<reduce_mean, mean_op>
{
    reduce_mean() {}
    reduce_mean(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
