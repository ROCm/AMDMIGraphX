#ifndef MIGRAPHX_GUARD_MARKER_HPP
#define MIGRAPHX_GUARD_MARKER_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// Marker is an interface to general marking functions, such as rocTX markers.

#else

<%
interface('marker',
           virtual('mark_start', ins_ref = 'instruction_ref', returns = 'void'),
           virtual('mark_start', prog = 'const program&', returns = 'void'),
           virtual('mark_stop', ins = 'instruction_ref', returns = 'void'),
           virtual('mark_stop', prog = 'const program&', returns = 'void')
        ) %>
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
