#ifndef MIGRAPH_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP
#define MIGRAPH_GUARD_RTGLIB_FWD_CONV_BATCHNORM_REWRITE_HPP

#include <string>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct program;

/**
 * Rewrite batchnorm to a multiply and add.
 */
struct fwd_conv_batchnorm_rewrite
{
    std::string name() const { return "fwd_conv_batchnorm_rewrite"; }
    void apply(program& p) const;
};

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
