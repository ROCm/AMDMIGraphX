#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>
#include <migraphx/cpu/parallel.hpp>
#include <migraphx/par_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace cpu {

struct context
{
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

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
