#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_SET_STREAM_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_SET_STREAM_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct program;

namespace gpu {

struct eliminate_set_stream
{
    std::string name() const { return "eliminate_set_stream"; }
    void apply(program& p) const;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
