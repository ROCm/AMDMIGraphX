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
           virtual('mark_start', ins_ref = 'instruction_ref', const = True, returns = 'void'),
           virtual('mark_start', prog = 'const program&', const = True, returns = 'uint64_t'),
           virtual('mark_stop', ins = 'instruction_ref', const = True, returns = 'void'),
           virtual('mark_stop', prog = 'const program&', const = True, returns = 'void')
        ) %>
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
