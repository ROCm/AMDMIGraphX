#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZATION_HPP

#include <string>
#include <vector>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

void quantize(program& prog, const std::vector<std::string>& ins_names);
void quantize(program& prog);

// insert the capture operator for the inputs of each operator to be quantized
// to int8
void capture_arguments(program& prog,
                       const std::vector<std::string>& ins_names,
                       std::function<void(std::size_t, std::vector<argument>)> func);
void capture_arguments(program& prog, const std::vector<std::string>& ins_names);
void capture_arguments(program& prog);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
