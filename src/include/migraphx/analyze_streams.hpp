#ifndef MIGRAPHX_GUARD_RTGLIB_ANALYZE_STREAMS_HPP
#define MIGRAPHX_GUARD_RTGLIB_ANALYZE_STREAMS_HPP

#include <migraphx/config.hpp>
#include <migraphx/stream_model.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct stream_race
{
    instruction_ref ins;
    // The instruction that should before
    instruction_ref before;
};

std::vector<stream_race> analyze_streams(const module& p, const stream_model& m);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
