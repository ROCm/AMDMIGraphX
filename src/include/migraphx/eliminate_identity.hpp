#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_IDENTITY_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_IDENTITY_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Remove identity instructions.
 */
struct eliminate_identity
{
    std::string name() const { return "eliminate_identity"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
