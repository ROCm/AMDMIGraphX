#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>
#include <migraphx/target.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

void quantize(program& prog, const std::vector<std::string>& ins_names);
void quantize(program& prog);

// insert the capture operator for the inputs of each operator to be quantized
// to int8
std::size_t capture_arguments(program& prog,
                              const std::vector<std::string>& ins_names,
                              const std::function<void(std::size_t, std::vector<argument>)>& func);
std::shared_ptr<std::vector<std::pair<float, float>>>
capture_arguments(program& prog, const target& t, const std::vector<std::string>& ins_names);
std::shared_ptr<std::vector<std::pair<float, float>>> capture_arguments(program& prog,
                                                                        const target& t);

void quantize_int8(program& prog,
                   const target& t,
                   std::vector<program::parameter_map>& calibration_args);
void quantize_int8(program& prog,
                   const target& t,
                   std::vector<program::parameter_map>& calibration_args,
                   const std::vector<std::string>& ins_names);
void quantize_int8(program& prog,
                   const std::vector<std::string>& ins_names,
                   const std::vector<std::pair<float, float>>& quant_params);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
