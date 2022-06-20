#ifndef MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DRIVER_VERIFY_HPP

#include "precision.hpp"
#include <migraphx/program.hpp>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

void verify_program(const std::string& name,
                    const program& p,
                    const target& t,
                    compile_options options     = compile_options{},
                    precision quantize          = precision::fp32,
                    const parameter_map& inputs = {},
                    double tolerance            = 100);
void verify_instructions(const program& prog,
                         const target& t,
                         compile_options options = compile_options{},
                         precision quantize      = precision::fp32,
                         double tolerance        = 80);
void verify_reduced_program(const program& p,
                            const target& t,
                            compile_options options     = compile_options{},
                            precision quantize          = precision::fp32,
                            const parameter_map& inputs = {},
                            double tolerance            = 80);

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx

#endif
