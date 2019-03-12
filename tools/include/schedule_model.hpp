#ifndef MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP
#define MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct operation;

#ifdef DOXYGEN

/// An interface for target-dependent model for the scheduler
struct schedule_model
{
    /// Get the number of concurrent instruction allowed
    std::size_t concurrency() const;
    /// Schedule a concurrent instruction
    void sched(program& p, instruction_ref ins, std::size_t n) const;
    // Insert necessary waits before an instruction
    void wait(program& p, instruction_ref ins, std::size_t wait_id) const;
    // Insert necessary records after an instruction
    void record(program& p, instruction_ref ins, std::size_t wait_id) const;
    /// Compute weights for an operation
    std::size_t weight(const operation& op) const;
};

#else

<%
interface('schedule_model',
    virtual('concurrency', returns='std::size_t', const=True),
    virtual('sched', p='program&', ins='instruction_ref', n='std::size_t', const=True),
    virtual('wait', p='program&', ins='instruction_ref', wait_id='std::size_t', const=True),
    virtual('record', p='program&', ins='instruction_ref', wait_id='std::size_t', const=True),
    virtual('weight', returns='std::size_t', op='const operation&', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
