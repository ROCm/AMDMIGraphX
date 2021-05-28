#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_QUANTIZELINEAR_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_QUANTIZELINEAR_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite quantizelinear to equivalent operators
 */
struct rewrite_quantizelinear
{
    std::string name() const { return "rewrite_quantizelinear"; }
    void apply(module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
