#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
struct program;

struct memory_coloring
{
    std::string allocation_op{};
    std::string name() const { return "memory coloring"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
