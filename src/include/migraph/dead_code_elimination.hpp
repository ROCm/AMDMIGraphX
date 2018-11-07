#ifndef MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP
#define MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

struct dead_code_elimination
{
    std::string name() const { return "dead_code_elimination"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
