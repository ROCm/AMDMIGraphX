#ifndef MIGRAPHX_GUARD_OPERATORS_OP_NORMALIZE_ATTRIBUTE_HPP
#define MIGRAPHX_GUARD_OPERATORS_OP_NORMALIZE_ATTRIBUTE_HPP

#include <migraphx/config.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

// different attributes
// 1) use_input(default)/use_output
// 2) use_rank(default)/use_len
// 3) clip_min(default)/not_clip_min
//   3.1) include_min(default)/exclude_min
// 4) clip_max(default)/not_clip_max
//   4.1) exclude_max(default)/include_max
// 5) normalize padding
enum class normalize_attribute
{
    use_len,
    use_output,
    clip_max,
    clip_min,
    include_max,
    include_min,
    normalize_padding
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
