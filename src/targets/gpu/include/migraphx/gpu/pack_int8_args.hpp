#ifndef MIGRAPHX_GUARD_RTGLIB_PACK_INT8_ARGS_HPP
#define MIGRAPHX_GUARD_RTGLIB_PACK_INT8_ARGS_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace gpu {

struct pack_int8_args
{
    std::string name() const { return "gpu::pack_int8_args"; }
    void apply(module& p) const;
    shape pack_int8_shape(const shape& s) const;
    std::vector<instruction_ref> pad_inputs(module& p, instruction_ref ins) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
