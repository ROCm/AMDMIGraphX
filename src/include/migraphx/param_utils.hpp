#ifndef MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::string param_name(std::size_t i, const std::string& prefix = "x");

void sort_params(std::vector<instruction_ref>& params);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_PARAM_UTILS_HPP
