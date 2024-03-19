#include <migraphx/param_utils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/builtin.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::string param_name(std::size_t i, const std::string& prefix)
{
    return prefix + std::to_string(i);
}

void sort_params(std::vector<instruction_ref>& params)
{
    std::sort(params.begin(), params.end(), by(std::less<>{}, [](instruction_ref ins) {
                  const auto& param = any_cast<const builtin::param&>(ins->get_operator());
                  return param.parameter;
              }));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx


