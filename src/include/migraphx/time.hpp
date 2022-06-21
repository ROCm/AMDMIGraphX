#ifndef MIGRAPHX_GUARD_RTGLIB_TIME_HPP
#define MIGRAPHX_GUARD_RTGLIB_TIME_HPP

#include <chrono>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct timer
{
    std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
    template <class Duration>
    auto record() const
    {
        auto finish = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<Duration>(finish - start).count();
    }
};

template <class Duration, class F>
auto time(F f)
{
    timer t{};
    f();
    return t.record<Duration>();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
