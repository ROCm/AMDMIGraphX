#ifndef MIGRAPHX_GUARD_OPERATORS_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_SUM_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sum_op
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

    auto output(const shape&) const
    {
        return [&](auto val) { return val; };
    }
};

struct reduce_sum : reduce_op<reduce_sum, sum_op>
{
    reduce_sum() {}
    reduce_sum(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
