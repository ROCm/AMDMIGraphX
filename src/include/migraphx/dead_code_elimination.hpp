#ifndef MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP
#define MIGRAPH_GUARD_RTGLIB_DEAD_CODE_ELIMINATION_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Remove instructions where the output is not used.
 */
struct dead_code_elimination
{
    std::string name() const { return "dead_code_elimination"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
