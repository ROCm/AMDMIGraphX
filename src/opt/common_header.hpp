#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#include <migraph/program.hpp>
#include <migraph/stringutils.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>

#include <set>
#include <list>
#include <vector>
#include <queue>

//#define MIGRAPH_DEBUG_OPT

#ifdef MIGRAPH_DEBUG_OPT
#define MIGRAPH_DEBUG(s) s
#else
#define MIGRAPH_DEBUG(s)
#endif // MIGRAPH_DEBUG_OPT

#endif // MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
