#ifndef MIGRAPHX_GUARD_MARKER_HPP
#define MIGRAPHX_GUARD_MARKER_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// Marker is an interface to general marking functions, such as rocTX markers.

#else

<%
 interface('marker',
           virtual('trace_ins', returns = 'void'),
           virtual('trace_prog', returns = 'void'),
           virtual('trace_ins_finish', returns = 'void'),
           virtual('trace_prog_finish', returns = 'void')) %>
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
