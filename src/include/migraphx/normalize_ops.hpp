#ifndef MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OPS_HPP
#define MIGRAPHX_GUARD_RTGLIB_NORMALIZE_OPS_HPP

#include <string>
#include <vector>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Process negative axis attributes of ops
 */

struct normalize_ops
{
    std::string name() const { return "normalize_ops"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
