#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_CONTIGUOUS_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Remove contiguous instructions by checking if the operator can use non-standard shapes.
 */
struct eliminate_contiguous
{
    std::string name() const { return "eliminate_contiguous"; }
    void apply(program& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
