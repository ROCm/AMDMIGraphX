#ifndef MIGRAPHX_GUARD_RTGLIB_COMMON_HEADER_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMMON_HEADER_HPP
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
inline namespace MIGRAPHX_INLINE_NS {

//#define MIGRAPHX_DEBUG_OPT

#ifdef MIGRAPHX_DEBUG_OPT
#define MIGRAPHX_DEBUG(s) s
#else
#define MIGRAPHX_DEBUG(s)
#endif // MIGRAPHX_DEBUG_OPT
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_RTGLIB_COMMON_HEADER_HPP
