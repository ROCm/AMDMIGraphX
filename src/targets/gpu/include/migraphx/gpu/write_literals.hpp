#ifndef MIGRAPHX_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP
#define MIGRAPHX_GUARD_RTGLIB_MIOPEN_WRITE_LITERALS_HPP

#include <migraphx/program.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

struct write_literals
{
    context* ctx = nullptr;
    std::string name() const { return "gpu::write_literals"; }

    void apply(program& p) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
