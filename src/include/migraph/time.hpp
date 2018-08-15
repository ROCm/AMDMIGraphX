#ifndef MIGRAPH_GUARD_RTGLIB_TIME_HPP
#define MIGRAPH_GUARD_RTGLIB_TIME_HPP

#include <chrono>

namespace migraph {

template <class Duration, class F>
auto time(F f)
{
    auto start = std::chrono::steady_clock::now();
    f();
    auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(finish - start).count();
}

} // namespace migraph

#endif
