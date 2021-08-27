#ifndef MIGRAPHX_GUARD_RTGLIB_QUANTIZE_UTIL_HPP
#define MIGRAPHX_GUARD_RTGLIB_QUANTIZE_UTIL_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

const std::vector<shape::type_t>& get_quantizable_type();
instruction_ref insert_quant_ins(module& modl,
                                 const instruction_ref& insert_loc,
                                 instruction_ref& ins,
                                 shape::type_t type,
                                 std::unordered_map<instruction_ref, instruction_ref>& map_ins,
                                 float scale = 1.0f,
                                 float shift = 0.0f);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
