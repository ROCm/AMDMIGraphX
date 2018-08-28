#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#include <migraph/program.hpp>
#include <migraph/stringutils.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/manage_ptr.hpp>

#include <set>
#include <list>
#include <vector>
#include <queue>

#define DEBUG_OPT

#ifdef DEBUG_OPT
#define DEBUG(s) s
#else
#define DEBUG(s)
#endif // DEBUG_OPT

#endif // MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
