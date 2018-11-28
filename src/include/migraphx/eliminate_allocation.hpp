#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Remove memory allocations. This will create a parameter which is the max of all memory used in
 * the program.
 */
struct eliminate_allocation
{
    std::string allocation_op{};
    std::size_t alignment = 32;
    std::string name() const { return "eliminate_allocation"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
