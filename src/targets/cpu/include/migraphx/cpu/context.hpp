#ifndef MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP
#define MIGRAPHX_GUARD_RTGLIB_CONTEXT_HPP

#include <migraphx/config.hpp>
#include <migraphx/cpu/dnnl.hpp>

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
};
#else
struct context
{
    void finish() const {}
};
#endif

} // namespace cpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
