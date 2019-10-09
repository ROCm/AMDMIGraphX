#ifndef MIGRAPHX_GUARD_RTGLIB_PAR_FOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_PAR_FOR_HPP

#include <thread>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct joinable_thread : std::thread
{
    template <class... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...) // NOLINT
    {
    }

    joinable_thread& operator=(joinable_thread&& other) = default;
    joinable_thread(joinable_thread&& other)            = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};

template <class F>
auto thread_invoke(std::size_t i, std::size_t tid, F f) -> decltype(f(i, tid))
{
    f(i, tid);
}

template <class F>
auto thread_invoke(std::size_t i, std::size_t, F f) -> decltype(f(i))
{
    f(i);
}

template <class F>
void par_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
        for(std::size_t i = 0; i < n; i++)
            thread_invoke(i, 0, f);
        ;
    }
    else
    {
        std::vector<joinable_thread> threads(threadsize);
// Using const here causes gcc 5 to ICE
#if(!defined(__GNUC__) || __GNUC__ != 5)
        const
#endif
            std::size_t grainsize = std::ceil(static_cast<double>(n) / threads.size());

        std::size_t work = 0;
        std::size_t tid  = 0;
        std::generate(threads.begin(), threads.end(), [=, &work, &tid] {
            auto result = joinable_thread([=] {
                std::size_t start = work;
                std::size_t last  = std::min(n, work + grainsize);
                for(std::size_t i = start; i < last; i++)
                {
                    thread_invoke(i, tid, f);
                }
            });
            work += grainsize;
            ++tid;
            return result;
        });
        assert(work >= n);
    }
}

template <class F>
void par_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize =
        std::min<std::size_t>(std::thread::hardware_concurrency(), n / min_grain);
    par_for_impl(n, threadsize, f);
}

template <class F>
void par_for(std::size_t n, F f)
{
    const int min_grain = 8;
    par_for(n, min_grain, f);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
