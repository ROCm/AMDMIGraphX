#ifndef MIGRAPHX_GUARD_RTGLIB_REWRITE_QUANTIZATION_HPP
#define MIGRAPHX_GUARD_RTGLIB_REWRITE_QUANTIZATION_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite quantization ops to equivalent operators
 */
struct rewrite_quantization
{
    std::string name() const { return "rewrite_quantization"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
