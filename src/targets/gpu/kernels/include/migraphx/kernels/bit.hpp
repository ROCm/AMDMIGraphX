#ifndef MIGRAPHX_GUARD_KERNELS_BIT_HPP
#define MIGRAPHX_GUARD_KERNELS_BIT_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

constexpr uint32_t get_bit(uint32_t x, uint32_t i)
{
    MIGRAPHX_ASSERT(i < 32);
    return (x >> i) & 1;
}

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
