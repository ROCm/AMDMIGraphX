#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP

// #define MIGRAPHX_DISABLE_OMP

#include <migraphx/config.hpp>
#ifdef MIGRAPHX_DISABLE_OMP
#include <migraphx/par_for.hpp>
#else
#include <omp.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

#ifdef MIGRAPHX_DISABLE_OMP

inline std::size_t max_threads() { return std::thread::hardware_concurrency(); }

template <class F>
void parallel_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
        f(std::size_t{0}, n);
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
        std::generate(threads.begin(), threads.end(), [=, &work] {
            auto result =
                joinable_thread([=]() mutable { f(work, std::min(n, work + grainsize)); });
            work += grainsize;
            return result;
        });
        // cppcheck-suppress unsignedLessThanZero
        assert(work >= n);
    }
}
#else

inline std::size_t max_threads() { return omp_get_max_threads(); }

template <class F>
void parallel_for_impl(std::size_t n, std::size_t threadsize, F f)
{
    if(threadsize <= 1)
    {
        f(std::size_t{0}, n);
    }
    else
    {
        std::size_t grainsize = std::ceil(static_cast<double>(n) / threadsize);
#pragma omp parallel for num_threads(threadsize) schedule(static, 1) private(grainsize, n)
        for(std::size_t tid = 0; tid < threadsize; tid++)
        {
            std::size_t work = tid * grainsize;
            f(work, std::min(n, work + grainsize));
        }
    }
}
#endif
template <class F>
void parallel_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize = std::min<std::size_t>(max_threads(), n / min_grain);
    parallel_for_impl(n, threadsize, f);
}

template <class F>
void parallel_for(std::size_t n, F f)
{
    const int min_grain = 8;
    parallel_for(n, min_grain, f);
}

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
