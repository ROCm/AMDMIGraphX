#ifndef MIGRAPHX_GUARD_OPERATORS_COMMON_HPP
#define MIGRAPHX_GUARD_OPERATORS_COMMON_HPP

#include <ostream>
#include <vector>
#include <migraphx/config.hpp>
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

// The pooling modes must correspond 1-1 to the operators defined for struct parse_pooling.
enum class pooling_mode
{
    avg,
    max
};

enum class roialign_mode
{
    avg,
    max
};

// indicate rnn computation direction
enum class rnn_direction
{
    forward,
    reverse,
    bidirectional,
};

std::ostream& operator<<(std::ostream& os, pooling_mode v);
std::ostream& operator<<(std::ostream& os, rnn_direction v);
std::ostream& operator<<(std::ostream& os, roialign_mode v);

std::string to_str(roialign_mode v);
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
