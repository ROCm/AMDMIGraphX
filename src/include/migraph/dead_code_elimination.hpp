#ifndef MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP
#define MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct dead_code_elimination
{
    std::string name() const { return "dead_code_elimination"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
