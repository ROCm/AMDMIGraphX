#ifndef MIGRAPHX_GUARD_OPERATORS_SCAN_INCLUSIVE_SUM_HPP
#define MIGRAPHX_GUARD_OPERATORS_SCAN_INCLUSIVE_SUM_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/argument.hpp>
#include <vector>
#include <migraphx/op/scan_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scan_inclusive_sum : scan_op<scan_inclusive_sum>
{
    scan_inclusive_sum() {}
    scan_inclusive_sum(std::vector<int64_t> ax) : scan_op(std::move(ax)) {}

    auto op() const
    {
        std::cout << "called self.op()" << std::endl;
        return [&](auto x, auto y) { x += y; };
    }

    /*auto output(const shape&) const
    {
        return [&](auto val) { return val; };
    }*/

};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
