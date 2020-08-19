#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct context
{
    value to_value() const
    {
        return value{};
    }

    void from_value(const value&)
    { }

    void finish() const {}
};

inline void migraphx_to_value(value& v, const context& ctx) { v = ctx.to_value(); }
inline void migraphx_from_value(const value& v, context& ctx) { ctx.from_value(v); }

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
