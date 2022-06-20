#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>
#include <migraphx/target.hpp>
#include <migraphx/program.hpp>
#include <migraphx/env.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

void quantize_fp16(program& prog, const std::vector<std::string>& ins_names = {"all"});

void quantize_int8(program& prog,
                   const target& t,
                   const std::vector<parameter_map>& calibration,
                   const std::vector<std::string>& ins_names = {"dot", "convolution"});

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
