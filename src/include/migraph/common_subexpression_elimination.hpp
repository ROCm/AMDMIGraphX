#ifndef MIGRAPH_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP
#define MIGRAPH_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {

struct program;

struct common_subexpression_elimination
{
    std::string name() const { return "common_subexpression_elimination"; }
    void apply(program& p) const;
};

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
