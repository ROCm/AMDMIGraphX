#ifndef MIGRAPHX_GUARD_OPERATORS_TUNE_AXIS_HPP
#define MIGRAPHX_GUARD_OPERATORS_TUNE_AXIS_HPP

#include <utility>
#include <cstdint>
#include <string>
#include <migraphx/errors.hpp>
#include <boost/algorithm/string.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline int tune_axis(const int n_dim, const int axis, const std::string& op_name = "OPERATOR")
{
    if(axis >= n_dim || abs(axis) > n_dim)
    {
        MIGRAPHX_THROW(boost::to_upper_copy<std::string>(op_name) + ": axis is out of range.");
    }
    return (axis < 0) ? axis + n_dim : axis;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
