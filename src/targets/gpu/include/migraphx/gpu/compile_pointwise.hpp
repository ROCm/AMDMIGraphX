#ifndef MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
#define MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
struct module;

namespace gpu {

struct context;
operation compile_pointwise(context& ctx,
                            const std::vector<shape>& inputs,
                            const std::string& lambda,
                            const std::string& preamble = "",
                            const int global_workitems = 1024,
                            const int local_workitems_per_CU = 256);  // todo: not clear where to #define LOCAL_THREADS to make it available here

operation compile_pointwise(context& ctx, const std::vector<shape>& inputs, module m);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_COMPILE_POINTWISE_HPP
