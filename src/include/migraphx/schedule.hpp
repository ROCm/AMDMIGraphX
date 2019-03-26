#ifndef MIGRAPHX_GUARD_RTGLIB_SCHEDULE_HPP
#define MIGRAPHX_GUARD_RTGLIB_SCHEDULE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/schedule_model.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Schedule instructions for concurrent execution
 */
struct schedule
{
    schedule_model model{};
    bool enable = true;
    std::string name() const { return "schedule"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
