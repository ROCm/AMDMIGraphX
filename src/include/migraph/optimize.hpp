#ifndef MIGRAPH_GUARD_RTGLIB_OPTIMIZE_HPP
#define MIGRAPH_GUARD_RTGLIB_OPTIMIZE_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {
struct program;

struct optimize
{
    std::string name() const { return "optimize"; }
    void apply(program& p) const;
};
    
} // namespace migraph


#endif
