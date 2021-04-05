#ifndef MIGRAPHX_GUARD_OPERATORS_SCAN_INCLUSIVE_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCAN_INCLUSIVE_SUM_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/op/prefix_scan_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct prefix_scan_sum : prefix_scan_op<prefix_scan_sum>
{
    prefix_scan_sum() {}
    prefix_scan_sum(int64_t ax) : prefix_scan_op(ax) {}
    prefix_scan_sum(int64_t ax, bool excl) : prefix_scan_op(ax, excl) {}
    prefix_scan_sum(int64_t ax, bool excl, bool rev) : prefix_scan_op(ax, excl, rev) {}

    auto op() const
    {
        return [](auto x, auto y) { return x + y; };
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
