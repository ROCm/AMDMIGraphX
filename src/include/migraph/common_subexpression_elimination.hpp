#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>

namespace migraph {

struct program;

struct common_subexpression_elimination
{
    std::string name() const { return "common_subexpression_elimination"; }
    void apply(program& p) const;
};

} // namespace migraph

#endif
