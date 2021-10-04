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
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// Marker is an interface to general marking functions, such as rocTX markers.

#else

<%
interface('marker',
           virtual('mark_range_start', range_id='std::size_t', returns = 'std::size_t'),
           virtual('mark_ins_start', log='std::string', returns = 'void'),
           virtual('mark_program_start', returns = 'void'),
           virtual('mark_range_finish', range_id='std::size_t', returns = 'void'),
           virtual('mark_ins_finish', returns = 'void'),
           virtual('mark_program_finish', returns = 'void'),
        ) %>
#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
