#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPES_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_TYPES_HPP

#include <migraphx/kernels/hip.hpp>

namespace migraphx {

using index_int = std::uint32_t;
using diff_int  = std::int32_t;

#define MIGRAPHX_DEVICE_CONSTEXPR constexpr __device__ __host__ // NOLINT

template <class T, index_int N>
using vec = T __attribute__((ext_vector_type(N)));

using half = _Float16;

} // namespace migraphx

#endif
