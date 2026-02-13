#ifndef ROCM_GUARD_ROCM_BIT_HPP
#define ROCM_GUARD_ROCM_BIT_HPP

#include <rocm/assert.hpp>
#include <rocm/config.hpp>
#include <rocm/type_traits.hpp>
#include <rocm/limits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <typename To,
          typename From,
          ROCM_REQUIRES(rocm::is_trivially_copyable<To>{} and
                        rocm::is_trivially_copyable<From>{} and sizeof(To) == sizeof(From))>
constexpr To bit_cast(From fr) noexcept
{
    return __builtin_bit_cast(To, fr);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countl_zero(T x) noexcept
{
    return __builtin_clzg(x, numeric_limits<T>::digits);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countl_one(T x) noexcept
{
    return countl_zero(T(~x));
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countr_zero(T x) noexcept
{
    return __builtin_ctzg(x, numeric_limits<T>::digits);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countr_one(T x) noexcept
{
    return countr_zero(T(~x));
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int popcount(T x) noexcept
{
    return __builtin_popcountg(x);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int bit_width(T x) noexcept
{
    return numeric_limits<T>::digits - countl_zero(x);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T bit_floor(T x) noexcept
{
    if(x != 0)
        return T(1) << (bit_width(x) - 1);
    return 0;
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T bit_ceil(T x) noexcept
{
    if(x <= 1)
        return 1;
    auto e = bit_width(T(x - 1));
    ROCM_ASSERT(e < numeric_limits<T>::digits);
    if constexpr(is_same<T, decltype(+x)>{})
        return T(1) << e;
    constexpr int offset_for_ub = numeric_limits<unsigned>::digits - numeric_limits<T>::digits;
    return T(1u << (e + offset_for_ub) >> offset_for_ub);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr bool has_single_bit(T x) noexcept
{
    return popcount(x) == 1;
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T rotl(T x, int s) noexcept
{
    const int n = numeric_limits<T>::digits;
    int r       = s % n;

    if(r == 0)
        return x;

    if(r > 0)
        return (x << r) | (x >> (n - r));

    return (x >> -r) | (x << (n + r));
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T rotr(T x, int s) noexcept
{
    const int n = numeric_limits<T>::digits;
    int r       = s % n;

    if(r == 0)
        return x;

    if(r > 0)
        return (x >> r) | (x << (n - r));

    return (x << -r) | (x >> (n + r));
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_BIT_HPP
