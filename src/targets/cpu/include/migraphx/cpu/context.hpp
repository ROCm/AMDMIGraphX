#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/cpu/parallel.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

#ifdef USE_DNNL
struct context
{
    dnnl::engine engine;
    dnnl::stream stream;

    context() : engine(dnnl::engine::kind::cpu, 0), stream(engine) {}
    void finish() const {}

    template <class F>
    void bulk_execute(std::size_t n, std::size_t min_grain, F f)
    {
        cpu::parallel_for(n, min_grain, f);
    }

    template <class F>
    void bulk_execute(std::size_t n, F f)
    {
        this->bulk_execute(n, 256, f);
    }
};
#else
struct context
{
    void finish() const {}

    template <class F>
    void bulk_execute(std::size_t n, std::size_t min_grain, F f)
    {
        const auto threadsize = std::min<std::size_t>(std::thread::hardware_concurrency(), n / min_grain);
        std::size_t grainsize = std::ceil(static_cast<double>(n) / threadsize);
        par_for(threadsize, 1, [&](auto tid) {
            std::size_t work = tid*grainsize;
            f(work, std::min(n, work+grainsize));
        });
    }

    template <class F>
    void bulk_execute(std::size_t n, F f)
    {
        this->bulk_execute(n, 256, f);
    }
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
