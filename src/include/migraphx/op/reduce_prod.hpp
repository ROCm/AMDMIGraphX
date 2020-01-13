#ifndef MIGRAPHX_GUARD_OPERATORS_REDUCE_PROD_HPP
#define MIGRAPHX_GUARD_OPERATORS_REDUCE_PROD_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_prod : reduce_op<reduce_prod>
{
    reduce_prod() {}
    reduce_prod(std::vector<int64_t> ax) : reduce_op(std::move(ax)) {}

    auto op() const
    {
        return [=](auto x, auto y) { return x * y; };
    }

    auto init() const { return one(); }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
