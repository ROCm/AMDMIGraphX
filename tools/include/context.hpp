#ifndef MIGRAPH_GUARD_CONTEXT_HPP
#define MIGRAPH_GUARD_CONTEXT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace migraph {

#ifdef DOXYGEN

/// A context is used to store internal data for a `target`. A context is
/// constructed by a target during compilation and passed to the operations
/// during `eval`.
struct context
{
};

#else

<%
interface('context')
%>

#endif

} // namespace migraph

#endif
