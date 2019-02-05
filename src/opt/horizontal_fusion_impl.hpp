#ifndef MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#define MIGRAPHX_GUARD_RTGLIB_HORIZONTAL_FUSION_IMPL_HPP
#include "common_header.hpp"
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct horizontal_fusion_impl
{
    horizontal_fusion_impl(program* p)
        : p_program(p)
    {
    }
    void run();
#ifdef MIGRAPHX_DEBUG_OPT
    void dump_program();
#endif
    program* p_program;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif
