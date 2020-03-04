#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_PREALLOCATE_PARAM_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_PREALLOCATE_PARAM_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct program;

namespace gpu {

struct preallocate_param
{
    std::string param{};
    context* ctx = nullptr;
    std::string name() const { return "preallocate_param"; }
    void apply(program& p) const;
};
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
