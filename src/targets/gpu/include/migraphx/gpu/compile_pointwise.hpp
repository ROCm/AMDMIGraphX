#ifndef MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct context;
operation
compile_pointwise(context& ctx,
                  const std::vector<shape>& inputs,
                  const std::string& lambda,
                  const std::string& preamble = "",
                  int global_workitems = 1024, // Global (total) work items. todo: testing indicates
                                               // values of global close to tensor size are best
                  int local_workitems = 256);  // Local work items per CU
operation compile_pointwise(context& ctx, const std::vector<shape>& inputs, module m);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
