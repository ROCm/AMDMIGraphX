#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#include <migraphx/program.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_config.hpp>
#include <migraphx/config.hpp>

#include <set>
#include <list>
#include <vector>
#include <queue>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

//#define MIGRAPH_DEBUG_OPT

#ifdef MIGRAPH_DEBUG_OPT
#define MIGRAPH_DEBUG(s) s
#else
#define MIGRAPH_DEBUG(s)
#endif // MIGRAPH_DEBUG_OPT

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif // MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
