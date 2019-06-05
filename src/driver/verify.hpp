#ifndef MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP

#include <migraphx/program.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

argument run_cpu(program p);
argument run_gpu(program p);
void verify_program(const std::string& name, program p, double tolerance = 100);
void verify_instructions(const program& prog, double tolerance = 80);
void verify_reduced_program(program p, double tolerance = 80);

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
