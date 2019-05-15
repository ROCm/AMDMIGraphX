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

void quantize_int8(program& prog, const std::vector<std::string>& ins_names);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
