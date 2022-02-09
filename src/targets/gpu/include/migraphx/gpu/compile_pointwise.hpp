#ifndef MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct context;
operation compile_pointwise(
    context& ctx,
    const std::vector<shape>& inputs,
    const std::string& lambda,
    size_t global_workitems,        // Global (total) work items. todo: testing indicates
                                    // values of global close to tensor size are best
    size_t local_workitems,         // Local work items per CU
    const std::string& preamble = "");       

// Overload of compile_pointwise without global, local.  It calculates global value at runtime
operation compile_pointwise(
    context& ctx,
    const std::vector<shape>& inputs,
    const std::string& lambda,
    const std::string& preamble = "");

operation compile_pointwise(context& ctx, const std::vector<shape>& inputs, module m);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
