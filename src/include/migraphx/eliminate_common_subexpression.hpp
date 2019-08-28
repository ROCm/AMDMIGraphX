#ifndef MIGRAPHX_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_COMMON_SUBEXPRESSION_ELIMINATION_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Remove identical instructions.
 */
struct eliminate_common_subexpression
{
    std::string name() const { return "eliminate_common_subexpression"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
