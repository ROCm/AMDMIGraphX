#ifndef MIGRAPHX_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP
#define MIGRAPHX_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

/**
 * Rewrite batchnorm to a multiply and add.
 */
struct rewrite_batchnorm
{
    std::string name() const { return "rewrite_batchnorm"; }
    void apply(module& p) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
