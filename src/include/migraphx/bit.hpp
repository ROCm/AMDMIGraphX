#ifndef MIGRAPHX_GUARD_MIGRAPHX_BIT_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_BIT_HPP

#include <migraphx/config.hpp>
#include <cstdint>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <unsigned int N>
constexpr unsigned int all_ones() noexcept
{
    return (1u << N) - 1u;
}

template <typename T>
constexpr int countl_zero(T value)
{
    unsigned int r = 0;
    for(; value != 0u; value >>= 1u)
        r++;
    return 8 * sizeof(value) - r;
}

constexpr std::uint64_t bit_ceil(std::uint64_t x) noexcept
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

constexpr std::uint32_t bit_ceil(std::uint32_t x) noexcept
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_BIT_HPP
