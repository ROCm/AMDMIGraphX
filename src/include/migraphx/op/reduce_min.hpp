#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_MIN_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_MIN_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_min : reduce_op<reduce_min>
{
    reduce_min() {}
    reduce_min(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x < y ? x : y; };
    }

    auto init() const { return highest(); }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
