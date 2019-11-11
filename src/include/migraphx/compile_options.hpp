#ifndef MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMPILE_OPTIONS_HPP

#include <migraphx/config.hpp>
#include <migraphx/tracer.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct compile_options
{
    bool offload_copy = true;
    tracer trace{};
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
