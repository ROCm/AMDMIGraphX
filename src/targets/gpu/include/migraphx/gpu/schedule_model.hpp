#ifndef MIGRAPHX_GUARD_RTGLIB_GPU_SCHEDULE_MODEL_HPP
#define MIGRAPHX_GUARD_RTGLIB_GPU_SCHEDULE_MODEL_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct operation;

namespace gpu {

struct schedule_model
{
    std::size_t streams = 0;
    std::size_t concurrency() const;
    void schedule_instruction(program& p, instruction_ref ins, std::size_t n) const;
    void wait(program& p,
              instruction_ref ins,
              std::size_t wait_on,
              const std::vector<std::size_t>& wait_for) const;
    std::size_t weight(const operation& op) const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
