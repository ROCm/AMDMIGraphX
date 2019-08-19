#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_POOLING_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_POOLING_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

/**
 * Rewrite pooling to reduce_mean
 */
struct rewrite_pooling
{
    std::string name() const { return "rewrite_pooling"; }
    void apply(program& prog) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
