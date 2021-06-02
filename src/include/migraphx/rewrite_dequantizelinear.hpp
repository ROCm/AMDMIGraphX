#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_DEQUANTIZELINEAR_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_DEQUANTIZELINEAR_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite dequantizelinear to equivalent operators
 */
struct rewrite_dequantizelinear
{
    std::string name() const { return "rewrite_dequantizelinear"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
