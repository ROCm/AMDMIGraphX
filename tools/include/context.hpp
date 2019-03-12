#ifndef MIGRAPHX_GUARD_CONTEXT_HPP
#define MIGRAPHX_GUARD_CONTEXT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>

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

<%
interface('context',
    virtual('finish', returns='void', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
