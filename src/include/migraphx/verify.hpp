#ifndef MIGRAPHX_GUARD_VERIFY_HPP
#define MIGRAPHX_GUARD_VERIFY_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>

#include <migraphx/float_equal.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// Compute the value of a range
template <class R>
using range_value = std::decay_t<decltype(*std::declval<R>().begin())>;

struct sum_fn
{
    template <class T, class U>
    auto operator()(T x, U y) const
    {
        return x + y;
    }
};
static constexpr sum_fn sum{};

struct max_fn
{
    template <class T>
    static T id(T x)
    {
        return x;
    }

    template <class T, class U>
    auto operator()(T x, U y) const
    {
        return x > y ? x : y;
    }
};
static constexpr max_fn max{};

namespace abs_diff_detail {
using std::fabs;
struct fn
{
    template <class T, class U>
    auto operator()(T x, U y) const
    {
        return fabs(x - y);
    }
};

} // namespace abs_diff_detail

static constexpr abs_diff_detail::fn abs_diff{};

struct not_finite_fn
{
    template <class T>
    bool operator()(T x) const
    {
        using std::isfinite;
        return not isfinite(x);
    }
};
static constexpr not_finite_fn not_finite{};

struct compare_mag_fn
{
    template <class T, class U>
    bool operator()(T x, U y) const
    {
        using std::fabs;
        return fabs(x) < fabs(y);
    }
};
static constexpr compare_mag_fn compare_mag{};

struct square_diff_fn
{
    template <class T, class U>
    double operator()(T x, U y) const
    {
        return (x - y) * (x - y);
    }
};
static constexpr square_diff_fn square_diff{};

template <class R1>
bool range_empty(R1&& r1)
{
    return r1.begin() == r1.end();
}

template <class R1>
auto range_distance(R1&& r1)
{
    return std::distance(r1.begin(), r1.end());
}

template <class R1>
bool range_zero(R1&& r1)
{
    return std::all_of(r1.begin(), r1.end(), [](auto x) { return float_equal(x, 0); });
}

template <class R1, class R2, class T, class Reducer, class Product>
T range_product(R1&& r1, R2&& r2, T state, Reducer r, Product p)
{
    return std::inner_product(r1.begin(), r1.end(), r2.begin(), state, r, p);
}

template <class R1, class R2, class Compare>
std::size_t mismatch_idx(R1&& r1, R2&& r2, Compare compare)
{
    auto p = std::mismatch(r1.begin(), r1.end(), r2.begin(), compare);
    return std::distance(r1.begin(), p.first);
}

template <class R1, class Predicate>
long find_idx(R1&& r1, Predicate p)
{
    auto it = std::find_if(r1.begin(), r1.end(), p);
    if(it == r1.end())
        return -1;
    else
        return std::distance(r1.begin(), it);
}

template <class R1, class R2>
double max_diff(R1&& r1, R2&& r2)
{
    return range_product(r1, r2, 0.0, max, abs_diff);
}

template <class R1, class R2, class T>
std::size_t mismatch_diff(R1&& r1, R2&& r2, T diff)
{
    return mismatch_idx(r1, r2, [&](auto x, auto y) {
        auto d = abs_diff(x, y);
        return float_equal(d, diff);
    });
}

template <class R1, class R2>
double rms_range(R1&& r1, R2&& r2)
{
    std::size_t n = range_distance(r1);
    if(n == range_distance(r2))
    {
        double square_difference = range_product(r1, r2, 0.0, sum_fn{}, square_diff);
        double mag1              = *std::max_element(r1.begin(), r1.end(), compare_mag);
        double mag2              = *std::max_element(r2.begin(), r2.end(), compare_mag);
        double mag =
            std::max({std::fabs(mag1), std::fabs(mag2), std::numeric_limits<double>::min()});
        return std::sqrt(square_difference) / (std::sqrt(n) * mag);
    }
    else
        return std::numeric_limits<range_value<R1>>::max();
}

template <class R1, class R2>
bool verify_range(R1&& r1, R2&& r2, double tolerance = 80, double* out_error = nullptr)
{
    double threshold = std::numeric_limits<range_value<R1>>::epsilon() * tolerance;
    auto error       = rms_range(r1, r2);
    // cppcheck-suppress uninitvar
    if(out_error != nullptr)
        *out_error = error;
    return error <= threshold;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
