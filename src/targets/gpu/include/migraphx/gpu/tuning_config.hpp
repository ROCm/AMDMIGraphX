#ifndef MIGRAPHX_GUARD_GPU_TUNING_CONFIG_HPP
#define MIGRAPHX_GUARD_GPU_TUNING_CONFIG_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct tuning_config
{
    value problem;
    std::vector<value> solutions;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_TUNING_CONFIG_HPP
