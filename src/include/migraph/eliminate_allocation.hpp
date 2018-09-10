#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_ALLOCATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

struct eliminate_allocation
{
    std::string allocation_op{};
    std::size_t alignment = 32;
    std::string name() const { return "eliminate_allocation"; }
    void apply(program& p) const;
};
} // namespace migraph

#endif
