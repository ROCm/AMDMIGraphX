#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
struct program;

/**
 * Remove memory allocations. It uses graph coloring to find memory allocations that can be reused.
 */
struct memory_coloring
{
    std::string allocation_op{};
    bool verify = false;
    std::string name() const { return "memory coloring"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
