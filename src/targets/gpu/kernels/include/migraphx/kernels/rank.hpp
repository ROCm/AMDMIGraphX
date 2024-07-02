#ifndef MIGRAPHX_GUARD_KERNELS_RANK_HPP
#define MIGRAPHX_GUARD_KERNELS_RANK_HPP

namespace migraphx {

template <int N>
struct rank : rank<N - 1>
{
};

template <>
struct rank<0>
{
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_RANK_HPP
