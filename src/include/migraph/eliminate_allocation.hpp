#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Remove memory allocations. This will create a parameter which is the max of all memory used in the program.
 */
struct eliminate_allocation
{
    std::string allocation_op{};
    std::size_t alignment = 32;
    std::string name() const { return "eliminate_allocation"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
