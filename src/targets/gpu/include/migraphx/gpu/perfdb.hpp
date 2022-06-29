#ifndef MIGRAPHX_GUARD_GPU_PERFDB_HPP
#define MIGRAPHX_GUARD_GPU_PERFDB_HPP

#include <migraphx/config.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/operation.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct problem_params
{
    operation op;
    std::vector<shape> inputs;
    shape output;
};

std::string get_mlir_perf_for_conv(const problem_params& pp);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_PERFDB_HPP
