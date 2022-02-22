#ifndef MIGRAPHX_GUARD_CONTEXT_HPP
#define MIGRAPHX_GUARD_CONTEXT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/any_ptr.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// A context is used to store internal data for a `target`. A context is
/// constructed by a target during compilation and passed to the operations
/// during `eval`.
struct context
{
    /// Wait for any tasks in the context to complete
    void finish() const;
};

#else

template <class T>
value to_value_context(const T&)
{
    return value{};
}

template <class T>
void from_value_context(T&, const value&)
{
}

template <class T>
any_ptr get_queue_context(T&)
{
    return {};
}

<%
 interface('context',
           virtual('to_value', returns = 'value', const = True, default = 'to_value_context'),
           virtual('from_value', v = 'const value&', default = 'from_value_context'),
           virtual('get_queue', returns = 'any_ptr', default = 'get_queue_context'),
           virtual('finish', returns = 'void', const = True)) %>

    inline void migraphx_to_value(value& v, const context& ctx)
{
    v = ctx.to_value();
}
inline void migraphx_from_value(const value& v, context& ctx) { ctx.from_value(v); }

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
