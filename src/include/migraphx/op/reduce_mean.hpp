#ifndef MIGRAPHX_GUARD_OPERATORS_MEAN_HPP
#define MIGRAPHX_GUARD_OPERATORS_MEAN_HPP

#include <migraphx/op/reduce_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct reduce_mean : reduce_op<reduce_mean, mean_op>
{
    reduce_mean() : reduce_op() { }
    reduce_mean(std::vector<int64_t> ax) : reduce_op(ax) 
    { }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
