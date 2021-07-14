#ifndef MIGRAPHX_GUARD_RTGLIB_REMOVE_QDQ_PAIRS_HPP
#define MIGRAPHX_GUARD_RTGLIB_REMOVE_QDQ_PAIRS_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Remove any remaining Q/DQ pairs after fusing and rewriting quantized operators
 */
struct remove_qdq_pairs
{
    std::string name() const { return "remove_qdq_pairs"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
