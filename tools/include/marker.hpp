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
#include <migraphx/instruction_ref.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// Marker is an interface to general marking functions, such as rocTX markers.

#else

<%
interface('marker',
           virtual('mark', st = 'std::string', returns = 'void'),
           virtual('range_start', st = 'std::string', returns = 'size_t'),
           virtual('range_stop', range_num = 'std::size_t', returns = 'void'),
           virtual('trace_ins_start', st = 'std::string', returns = 'void'),
           virtual('trace_ins_end', returns = 'void')
        ) %>
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
