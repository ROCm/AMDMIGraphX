#ifndef MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP

#include <migraph/argument.hpp>
#include <migraph/literal.hpp>
#include <random>

namespace migraph {

template <class T, MIGRAPH_REQUIRES(std::is_floating_point<T>{})>
constexpr T normalize(unsigned long z)
{
    if(z == 0)
        return 0;
    const auto max     = 2048;
    const double range = max / 2; // NOLINT
    double result      = (z % max) / range;
    result -= 1;
    return result;
}

template <class T, MIGRAPH_REQUIRES(std::is_signed<T>{} and not std::is_floating_point<T>{})>
constexpr T normalize(unsigned long z)
{
    const auto max      = std::numeric_limits<T>::max();
    const auto half_max = max / 2;
    return half_max - (z % max);
}

template <class T, MIGRAPH_REQUIRES(not std::is_signed<T>{} and std::is_integral<T>{})>
constexpr T normalize(unsigned long z)
{
    const auto max = std::numeric_limits<T>::max();
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
std::vector<T> generate_tensor_data(const migraph::shape& s, unsigned long seed = 0)
{
    std::vector<T> result(s.elements());
    std::generate(result.begin(), result.end(), xorshf96_generator<T>{seed});
    // std::generate(result.begin(), result.end(), [&]{ return seed % 7; });
    // std::generate(result.begin(), result.end(), []{ return 1; });
    return result;
}

argument generate_argument(shape s, unsigned long seed = 0);

literal generate_literal(shape s, unsigned long seed = 0);

literal abs(literal l);

} // namespace migraph

#endif
