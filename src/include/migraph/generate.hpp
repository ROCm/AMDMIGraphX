#ifndef MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_GENERATE_HPP

#include <migraph/argument.hpp>
#include <migraph/literal.hpp>
#include <random>

namespace migraph {

template <class T, MIGRAPH_REQUIRES(std::is_floating_point<T>{})>
T normalize(unsigned long z)
{
    if(z == 0)
        return 0;
    return (2.0 / z) - 1.0;
}

template <class T, MIGRAPH_REQUIRES(std::is_signed<T>{} and not std::is_floating_point<T>{})>
T normalize(unsigned long z)
{
    const auto max      = std::numeric_limits<T>::max();
    const auto half_max = max / 2;
    return half_max - (z % max);
}

template <class T, MIGRAPH_REQUIRES(not std::is_signed<T>{} and std::is_integral<T>{})>
T normalize(unsigned long z)
{
    const auto max = std::numeric_limits<T>::max();
    return z % max;
}

template <class T>
struct xorshf96_generator
{
    unsigned long x = 123456789;
    unsigned long y = 362436069;
    unsigned long z = 521288629;

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
std::vector<T> generate_tensor_data(const migraph::shape& s, std::mt19937::result_type)
{
    std::vector<T> result(s.elements());
    std::generate(result.begin(), result.end(), xorshf96_generator<T>{});
    return result;
}

argument generate_argument(shape s, std::mt19937::result_type seed = 0);

literal generate_literal(shape s, std::mt19937::result_type seed = 0);

} // namespace migraph

#endif
