#ifndef MIGRAPHX_GUARD_KERNELS_BIT_HPP
#define MIGRAPHX_GUARD_KERNELS_BIT_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

constexpr uint64_t bit_ceil(uint64_t x)
{
    if (x <= 1)
        return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

constexpr uint32_t bit_ceil(uint32_t x)
{
    if (x <= 1)
        return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_BIT_HPP
