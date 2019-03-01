#include <migraphx/gpu/schedule_model.hpp>
#include <migraphx/program.hpp>
#include <migraphx/operation.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

std::size_t schedule_model::concurrency() const { return n; }
void schedule_model::schedule_instruction(program& p, instruction_ref ins, std::size_t n) const {}
void schedule_model::wait(program& p,
                          instruction_ref ins,
                          std::size_t wait_on,
                          const std::vector<std::size_t>& wait_for) const
{
}
std::size_t schedule_model::weight(const operation& op) const { return 1; }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx