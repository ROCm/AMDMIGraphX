#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_CPU_PARALLEL_HPP

#include <migraphx/config.hpp>
#if USE_DNNL
#include <omp.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

#if USE_DNNL
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
        #pragma omp parallel num_threads(threadsize)
        {
            std::size_t tid = omp_get_thread_num();
            std::size_t work = tid*grainsize;
            f(work, std::min(n, work+grainsize));
        }
    }
}

template <class F>
void parallel_for(std::size_t n, std::size_t min_grain, F f)
{
    const auto threadsize =
        std::min<std::size_t>(omp_get_num_threads(), n / min_grain);
    parallel_for_impl(n, threadsize, f);
}

template <class F>
void parallel_for(std::size_t n, F f)
{
    const int min_grain = 8;
    parallel_for(n, min_grain, f);
}
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
