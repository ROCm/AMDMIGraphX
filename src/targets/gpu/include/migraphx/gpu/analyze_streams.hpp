#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_ANALYZE_STREAMS_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_ANALYZE_STREAMS_HPP

#include <migraphx/config.hpp>
#include <migraphx/analyze_streams.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::vector<stream_race> analyze_streams(const program& p);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
