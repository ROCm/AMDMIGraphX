#ifndef MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP
#define MIGRAPH_GUARD_RTGLIB_MEMORY_COLORING_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

struct memory_coloring
{
    std::string allocation_op{};
    std::string name() const { return "memory coloring"; }
    void apply(program& p) const;
};
} // namespace migraph

#endif
