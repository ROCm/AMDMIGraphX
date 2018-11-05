#ifndef MIGRAPH_GUARD_RTGLIB_RANK_HPP
#define MIGRAPH_GUARD_RTGLIB_RANK_HPP

#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
