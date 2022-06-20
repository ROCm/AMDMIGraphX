#ifndef MIGRAPHX_GUARD_RTGLIB_PERF_HPP
#define MIGRAPHX_GUARD_RTGLIB_PERF_HPP

#include <migraphx/program.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

parameter_map
fill_param_map(parameter_map& m, const program& p, const target& t, bool offload = false);
parameter_map create_param_map(const program& p, const target& t, bool offload = false);

parameter_map fill_param_map(parameter_map& m, const program& p, bool gpu);
parameter_map create_param_map(const program& p, bool gpu = true);
target get_target(bool gpu);
void compile_program(program& p, bool gpu = true);

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
