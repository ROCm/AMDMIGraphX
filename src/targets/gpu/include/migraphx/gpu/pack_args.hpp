#ifndef MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_PACK_ARGS_HPP

#include <migraphx/config.hpp>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::vector<char> pack_args(const std::vector<std::pair<std::size_t, void*>>& args);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
