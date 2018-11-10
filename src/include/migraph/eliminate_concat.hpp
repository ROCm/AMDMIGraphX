#ifndef MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP
#define MIGRAPH_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP

#include <string>
#include <migraph/instruction_ref.hpp>
#include <migraph/concat_opt.hpp>
#include <migraph/config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Remove concat operators by having each operator can write to different chunk of memory.
 */
struct eliminate_concat
{
    concat_optimization concat_opt;
    std::string name() const { return "eliminate_concat"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
