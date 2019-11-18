#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_GENERATE_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_GENERATE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/type_traits.hpp>
#include <migraphx/config.hpp>
#include <random>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class T, MIGRAPHX_REQUIRES(is_floating_point<T>{})>
constexpr T normalize(unsigned long z)
{
    if(z == 0)
        return T(0);
    const auto max     = 32;
    const double range = max / 2; // NOLINT
    double result      = double(z % max) / range;
    result -= 1;
    return T(result);
}

template <class T, MIGRAPHX_REQUIRES(is_signed<T>{} and not is_floating_point<T>{})>
constexpr T normalize(unsigned long z)
{
    const auto max      = std::numeric_limits<T>::max() / 64;
    const auto half_max = max / 2;
    return half_max - (z % max);
}

template <class T, MIGRAPHX_REQUIRES(not is_signed<T>{} and std::is_integral<T>{})>
constexpr T normalize(unsigned long z)
{
    const auto max = std::numeric_limits<T>::max() / 64;
    return z % max;
}

template <class T>
struct xorshf96_generator
{
    unsigned long x = 123456789;
    unsigned long y = 362436069;
    unsigned long z;

    xorshf96_generator(unsigned long seed = 0) : z(521288629ULL ^ seed) {}

    constexpr T operator()() noexcept
    {
        x ^= x << 16U;
        x ^= x >> 5U;
        x ^= x << 1U;

        unsigned long t = x;
        x               = y;
        y               = z;
        z               = t ^ x ^ y;

        return normalize<T>(z);
    }
};

template <class T>
struct xorshift_generator
{
    unsigned long x;

    xorshift_generator(unsigned long seed = 0) : x(521288629ULL ^ seed) {}

    constexpr T operator()() noexcept
    {
        x ^= x >> 12U;
        x ^= x << 25U;
        x ^= x >> 27U;
        return normalize<T>(x * 0x2545F4914F6CDD1D);
    }
};

template <class T>
auto generate_tensor_data(const migraphx::shape& s, unsigned long seed = 0)
{
    auto result = make_shared_array<T>(s.elements());
    std::generate(result.get(), result.get() + s.elements(), xorshf96_generator<T>{seed});
    return result;
}

template <class T>
auto fill_tensor_data(const migraphx::shape& s, unsigned long value = 0)
{
    auto result = make_shared_array<T>(s.elements());
    std::generate(result.get(), result.get() + s.elements(), [=] { return value; });
    return result;
}

argument fill_argument(shape s, unsigned long value = 0);

argument generate_argument(shape s, unsigned long seed = 0);

literal generate_literal(shape s, unsigned long seed = 0);

literal abs(literal l);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
