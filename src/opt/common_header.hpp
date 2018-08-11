#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#define DEBUG_OPT

#ifdef DEBUG_OPT
#define DEBUG(s) s
#else
#define DEBUG(s)
#endif // DEBUG_OPT

#endif  // MIGRAPH_GUARD_RTGLIB_COMMON_HEADER_HPP
