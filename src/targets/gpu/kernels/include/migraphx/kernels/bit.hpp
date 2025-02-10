#ifndef MIGRAPHX_GUARD_KERNELS_BIT_HPP
#define MIGRAPHX_GUARD_KERNELS_BIT_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

constexpr bool get_bit(uint32_t x, uint32_t i) noexcept
{
    MIGRAPHX_ASSERT(i < 32);
    return ((x >> i) & 1) != 0;
}

constexpr uint64_t bit_ceil(uint64_t x) noexcept
{
    if(x <= 1)
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

constexpr uint32_t bit_ceil(uint32_t x) noexcept
{
    if(x <= 1)
        return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

constexpr uint32_t popcount(uint32_t x) noexcept { return __popc(x); }

constexpr uint32_t popcount(uint64_t x) noexcept { return __popcll(x); }

constexpr uint32_t countr_zero(uint32_t x) noexcept
{
    // popcount(~(x | −x))
    return __builtin_ctz(x);
}

constexpr uint32_t countr_zero(uint64_t x) noexcept { return __builtin_ctzll(x); }

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_BIT_HPP
