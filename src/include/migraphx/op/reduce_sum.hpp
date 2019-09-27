#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_SUM_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_sum : reduce_op<reduce_sum>
{
    reduce_sum() {}
    reduce_sum(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x + y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
