#ifndef MIGRAPHX_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP
#define MIGRAPHX_GUARD_RTGLIB_ELIMINATE_CONCAT_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/concat_opt.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
