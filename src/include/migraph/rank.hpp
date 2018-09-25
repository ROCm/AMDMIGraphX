#ifndef MIGRAPH_GUARD_RTGLIB_RANK_HPP
#define MIGRAPH_GUARD_RTGLIB_RANK_HPP

namespace migraph {

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

} // namespace migraph

#endif
