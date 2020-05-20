#ifndef MIGRAPHX_GUARD_OPERATORS_COMMON_HPP
#define MIGRAPHX_GUARD_OPERATORS_COMMON_HPP

#include <array>
#include <migraphx/operation.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/config.hpp>
#include <cmath>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

enum padding_mode_t
{
    default_, // NOLINT
    same,
    valid
};

// indicate rnn computation direction
enum class rnn_direction
{
    forward,
    reverse,
    bidirectional,
};

enum reduce_mode_t
{
    sum,
    mean,
    max_
};

std::ostream& operator<<(std::ostream& os, rnn_direction v);

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
